# RAQ-KV: Araştırma Planı

**Sürüm:** v0.1 · **Tarih:** 2026-09-17 · **Durum:** Kod yazılmadı; yalnızca fizibilite testi yapıldı (bkz. §0)

> Her deney için sorulacak soru: *"Bu deney makaledeki hangi iddiayı kanıtlamak için var?"*
> Bu sorunun cevabı `docs/claims.md` içindeki iddia ↔ deney ↔ figür matrisinde tutulacak.

---

## 0. Planın dayandığı ölçülmüş bulgular

Bu plan ezberden yazılmadı. Aşağıdaki bulgular 2026-09-17'de bu makinede ölçüldü.

**Donanım/ortam:** RTX 5060 Ti 16 GB (Blackwell, compute capability 12.0), sürücü 576.88, torch 2.11.0+cu128 (CUDA çalışıyor), Python 3.12.10, 63 GB RAM, ~307 GB boş disk, Windows 10. PyPI'daki güncel transformers sürümü **5.17.0**. Transformers 4.x ile arasında Cache API'si büyük ölçüde değişti.

**KV-cache fizibilite testi** (transformers 5.17.0; rastgele ağırlıklı küçük bir Qwen3 modeli, GPU, bf16):

| Soru | Sonuç | Plana etkisi |
|---|---|---|
| KV nerede tutuluyor? | `DynamicCache.layers[i].keys / .values`, şekil `[B, H_kv, T, d_head]` | Adres uzayı bu yerleşim üzerine kurulur |
| Analitik KV byte sayısı gerçek tensör byte sayısına eşit mi? | Eşit (15360 = 15360) | Bit hesabı FP16'da gerçek ölçümle doğrulanabilir |
| Tek geçişteki logits, prefill+decode logits ile aynı mı? | max\|Δ\| = 0.0 | İki fazlı protokol geçerli; gerçek modelde M0'da tekrar ölçülecek |
| Cache tensörü yerinde değiştirilirse decode etkileniyor mu? | Evet, max\|Δlogit\| = 0.22 | Quantization ve fault injection cache üzerinden uygulanabilir |
| Token budandıktan sonra ne oluyor? | `get_seq_length()` budanmış uzunluğu döndürüyor (20 → 17); maske bu uzunluktan otomatik türetiliyor; `position_ids` açıkça verilirse decode doğru. Ortadan budama + çok token'lı chunk, maskeli tam cache ile aynı (max\|Δ\| = 4.2·10⁻⁷) | **RoPE pozisyonu açıkça verilmezse pozisyon sessizce kayar** (max\|Δ\| = 0.30) |
| Katman düzeyinde `update()` yakalanabiliyor mu? | Katman başına hem prefill'de hem decode'da çağrılıyor | M5'te Cache/Layer subclass yazmak mümkün |
| Hazır quantized cache var mı? | `QuantizedCache`, `QuantizedLayer`, `QuantoQuantizedLayer`, `HQQQuantizedLayer` | Referans olarak kullanılır; bit düzeyinde erişim vermediği için kendi katmanımız gerekir |
| `sdpa` ile `output_attentions=True` | **Yalnızca uyarı verip boş liste `[]` döndürüyor** | Attention importance sessizce boş kalabilir; açık Q·K hesabı ve assert zorunlu |
| Qwen3-1.7B, non-thinking + tools | `enable_thinking=False` boş `<think></think>` ekliyor; tool çağrısı `<tool_call>{json}</tool_call>` | Kısa ve deterministik çıktı; tool argümanları AST ile değerlendirilebilir |

---

## A. Araştırma problemi

LLM ajanları uzun oturumlarda çalışır: system prompt, tool şemaları, geçmiş tool sonuçları ve konuşma birikir. Bu sırada bellekte en hızlı büyüyen bileşen KV-cache'tir. KV-cache'in maliyetini düşürmek için birbirinden bağımsız üç kaldıraç var:

1. **Retention:** hangi token'ın KV'si tutulacak (pruning/eviction).
2. **Precision:** tutulan KV kaç bitle saklanacak (quantization).
3. **Protection:** saklanan bitler hangi hata koruma koduyla korunacak (ECC).

Literatür bu kaldıraçları ayrı ayrı optimize ediyor ve genellikle belleğin hatasız olduğunu varsayıyor. Oysa üçü de **aynı kaynağı tüketiyor: bellekteki bitleri.** Bir ECC check biti, bir precision biti ve tutulan bir token aynı bütçeden harcanıyor.

**Formel problem.** Her KV grubu g (token t × layer l × KV-head h × {K,V}) için bir karar verilir: d_g = (r_g, b_g, e_g), burada r ∈ {KEEP, PRUNE, (COMPRESS)}, b ∈ {2,4,8,16} ve e ∈ {NONE, PARITY, SECDED, (BCH)}. Toplam maliyet:

```
C(d) = Σ_g 1[r_g ≠ PRUNE] · ( n_g·b_g  +  meta(b_g)  +  ecc(e_g, n_g·b_g) )  +  M_policy(d)
```

`meta` quantization metadata'sını (scale ve zero-point), `M_policy` ise politika metadata'sını (bit-width haritası, ECC mod haritası, tutulan token indeksleri) içerir.

```
max_d  TaskQuality(d)
s.t.   C(d) ≤ B                     (eşit toplam bit bütçesi)
       SDC(d; BER, fault model) ≤ ε
```

Kullanıcının tanımladığı çok amaçlı objective (latency, energy, memory, token, ECC overhead), bu problemin Lagrange biçimidir. Pareto analizi farklı B ve ε değerleri taranarak yapılır.

**Bilimsel soru.** KV-cache'teki hassasiyet (layer × KV-head × token × K/V × bit pozisyonu) yeterince heterojen mi? Eğer heterojense, hassasiyeti belirleyen üç sinyal (semantik önem, quantization hassasiyeti, hata kırılganlığı) aynı bölgeleri mi işaret ediyor, yoksa ayrı bilgi mi taşıyor?

**Tehdit modeli (kapsam):**
- **Kapsam içi:** Prefill'den sonra bellekte saklanan KV'de ve quantization metadata'sında oluşan bit hataları. Kaynak olarak soft error ile enerji tasarrufu için gevşetilmiş refresh ve düşürülmüş voltaj (Flikker tarzı) ele alınır.
- **Kapsam dışı (ilk faz):** ağırlıklardaki hatalar, hesaplama (logic) hataları, Rowhammer gibi kasıtlı saldırılar, eğitim.
- **Neden KV-cache:** KV bir kez yazılır, sonraki her decode adımında okunur. Saklanan tek bir hatalı değer, sonraki bütün token'ların attention hesabına girer (*persistence amplification*). Tek adımlık bir aktivasyon hatasında böyle bir birikim olmaz.
- **BER gerçekçiliği:** Sahadaki BER çok düşük olduğu için deneyler hızlandırılmış hata enjeksiyonuyla (accelerated fault injection) yapılır. Sonuçlar iki ayrı nicelik olarak raporlanır: (i) koşullu kırılganlık P(SDC | s konumunda hata) ve (ii) BER'e bağlı beklenen SDC eğrisi.

---

## B. Ana yenilik

### B.1 Yakın literatür ve boşluk

| Hat | Temsilci çalışmalar | Ne yapıyor | Neyi yapmıyor |
|---|---|---|---|
| KV token seçimi/eviction | H2O, SnapKV, StreamingLLM, PyramidKV | Önem skoruna göre token tutar | Belleği hatasız varsayar; precision sabittir |
| Prompt sıkıştırma | LLMLingua, LongLLMLingua | Giriş token sayısını azaltır | KV, precision ve güvenilirlikle ilgilenmez |
| KV quantization | KIVI, KVQuant, GEAR | Düşük bitli KV | Genelde tek tip bit kullanır; hatasız bellek varsayar |
| Önem-farkındalıklı karma hassasiyetli KV | ZipCache, MiKV | Önemli token'a daha fazla bit verir | Hatasız bellek varsayar; ECC yok |
| DNN hata dayanıklılığı ve seçici koruma | Ares, PyTorchFI ve seçici ECC çalışmaları | Fault injection; katman/bit hassasiyeti | Çoğunlukla ağırlık ve aktivasyonlara bakar; token semantiği ve retention kararı yok |

### B.2 Katkının çekirdeği

1. **Ortak bit para birimi.** Retention, precision ve protection kararları tek bir "korunmuş bit" bütçesinden karşılanır. Karar, bit başına marjinal faydaya göre verilir. **Neden önemli:** Tek başına seçici ECC'nin kazanç tavanı düşüktür. INT4 veri üzerinde SECDED(72,64) kullanıldığında overhead %12.5'tir; bütün ECC kaldırılsa bile en fazla %12.5 tasarruf edilir. Güçlü iddia, bu bitlerin kritik bölgelerde precision'a (INT4→INT8) veya retention'a aktarılmasından çıkar.
2. **Üç sinyalin ilişkisinin ölçülmesi.** Corr(QuantSens, FaultSens) ve Corr(Semantic, FaultImpact) değerleri KV büyüklüğü (outlier) confound'u kontrol edilerek hesaplanır. Sonuç hangi yönde çıkarsa çıksın yayımlanabilir (bkz. I).
3. **Ajan düzeyinde SDC.** Token düzeyindeki metriklerin (PPL, KL), tool argümanlarındaki ve durum taşıyan token'lardaki hataların görev üzerindeki etkisini hafife alıp almadığı gösterilir.

### B.3 Alt hipotez: tespit edilen hata = zorla eviction

Tespit edilip düzeltilemeyen bir hatada (DUE) ilgili token maskelenebilir, yani budanmış gibi davranılabilir. Bu durumda **PARITY (yalnız tespit) + önem-farkındalıklı retention**, düşük önemli bölgelerde SECDED'e yakın sonuç verebilir ve daha az bit harcar. ECC ekseni ile retention ekseni böylece doğrudan birbirine bağlanır.

### B.4 Codeword granülerliği de bir tahsis boyutudur

| SECDED kodu | Check bit / veri biti | Overhead |
|---|---|---|
| (13,8) | 5/8 | %62.5 |
| (22,16) | 6/16 | %37.5 |
| (39,32) | 7/32 | %21.9 |
| (72,64) | 8/64 | %12.5 |

Codeword küçüldükçe veri başına düzeltme kapasitesi artar ama overhead da büyür.

### B.5 Uyarı

Literatür bilgim 2026 ortasına kadar uzanıyor ve bu alan hızlı ilerliyor. **M5'ten önce 2024–2026 dönemini kapsayan sistematik bir literatür taraması zorunlu.** Anahtar kelimeler: *KV cache soft error, LLM inference reliability, approximate memory LLM, selective ECC LLM, KV cache bit flip*. Çok yakın bir çalışma bulunursa, novelty iddiası B.2-(2) ve B.2-(3)'e kaydırılır.

---

## C. En büyük 5 teknik risk

### R1: Ölçüm protokolünün geçerliliği (KV erişimi)

- **Risk:** Tek geçişlik (single forward) bir PPL ölçümünde KV aynı geçişte hem üretilip hem kullanılır. Bu yüzden KV quantization ve fault'un **hiçbir etkisi görünmez**. Budamadan sonra `get_seq_length` kısalır; HF yeni token'lara yanlış RoPE pozisyonu verir. `sdpa` attention ağırlıklarını sessizce boş döndürür. Transformers 5.x Cache API'si sürümler arasında değişir.
- **Kanıt:** §0'daki fizibilite testi bu davranışların dördünü de doğruladı.
- **Önlem:**
  - İki fazlı protokol: context prefill → KV snapshot → dönüşüm (budama, quantization, ECC, fault) → sorgu veya continuation'ın cache üzerinden skorlanması.
  - **Tek kontrol edilen pozisyon argümanı: `position_ids`** (düzeltme, 2026-09-17). transformers 5.17 kaynak kodunda `Qwen3Model.forward` artık `cache_position` almıyor. İlk fizibilite testinde verilen `cache_position` `**kwargs` içinde sessizce yutuldu. Kaynak kod ve ikinci test şunu gösteriyor:
    - **Maske boyutu ve sorgu offset'i** cache'in fiziksel (budanmış) uzunluğundan otomatik türetilir: `get_mask_sizes`, `get_query_offset` = `get_seq_length`. Retained token'ların sırası korunduğu sürece budama kendi içinde tutarlıdır.
    - **RoPE** yalnızca `position_ids`'ten beslenir. Budanmış cache'te bu değer orijinal pozisyon olarak **açıkça** verilmelidir.
    - **Doğrulama (ikinci test, küçük model, float32, sdpa ve eager):** Ortadan budanmış cache + açık `position_ids` + 5 token'lık chunk + 3 adım greedy, pozisyonları 2D `attention_mask` ile gizlenmiş tam cache ile aynı sonucu verdi: max|Δ| = 4.2·10⁻⁷, greedy token'lar aynı. `position_ids` verilmediğinde max|Δ| = 0.30 çıktı; sessiz RoPE kayması doğrulandı.
  - **Değiştirilmiş (mutated) cache üzerinde `model.generate()` kullanılmaz.** `generate()` `position_ids`'i cache uzunluğundan türetir; budanmış cache'te RoPE hata vermeden bozulur. Bunun yerine kendi yazdığımız greedy döngü (`evaluation/protocol.py`) M0'da hazırlanır; GSM8K de bu döngüyle koşulur.
  - **Budama eşdeğerliği için referans:** Tam cache + budanan pozisyonlarda 0 olan 2D `attention_mask`. Cache geometrisini değiştirmeden aynı hesabı verir.
  - **Batch kuralı.** Değiştirilmiş cache yollarında padding ile batch yapılmaz. Batch yalnızca **aynı örneğin, aynı tutulan uzunluğa sahip varyantları** arasında yapılır (ör. aynı snapshot üzerinde farklı fault denemeleri). Bu koşulda padding olmadığı için maske sorunu çıkmaz ve M2 kampanyası yine hızlanır. Golden geçişlerde ise normal batch serbesttir.
  - Attention importance, gözlem penceresi üzerinde açık Q·K hesabıyla bulunur ve boş sonuç assert ile yakalanır.
  - transformers **5.17.0** sürümüne sabitlenir; `scripts/probe_cache_api.py` her sürüm yükseltmesinde tekrar koşulur.
- **Tehdit ettiği iddia:** Tümü. Protokol yanlışsa bütün Δ ölçümleri geçersiz olur.

### R2: Simülasyon sonucunun donanım iddiası gibi okunması

- **Risk:** Fake-quant (quantize→dequantize, hesaplama BF16'da) VRAM'de gerçek tasarruf sağlamaz. Python'daki dequantize, ECC decode ve bit işlemleri FP16 baseline'dan **yavaştır**. "Daha az bellek, daha düşük gecikme" iddiası bu ölçümlere dayanırsa hakem reddeder.
- **Önlem:**
  - **Bellek:** `budget/bit_accounting.py` ile analitik hesaplanır. Formül M0'da FP16 için gerçek tensör byte sayısıyla ve `torch.cuda.memory_allocated` farkıyla doğrulanır.
  - **Gecikme:** Yalnızca gerçekten ölçülebilen durumlarda raporlanır (FP baseline, token budamanın attention maliyetine etkisi). Quantization ve ECC gecikmesi ile enerji, ileride donanım modeline (RTL sentezi, bellek simülatörü) bırakılır.
  - Makalede "simulated" ve "measured" ayrımı açıkça yazılır.
- **Tehdit ettiği iddia:** Verimlilik iddiaları, §20'deki SPEED ve HARDWARE metrikleri.

### R3: Fault uzayının patlaması, nadir olay istatistiği ve fault modelinin gerçekçiliği

- **Fault uzayı:** Qwen3-1.7B, 1K token ve INT4 için bir örnekte 28×2×8×128×1000×4 ≈ **2.3·10⁸ veri biti** vardır (BF16'da ≈ 9.2·10⁸).
- **Nadir olay istatistiği:** p≈0.05'lik bir SDC oranını ±0.01 hassasiyetle (%95 güven) ölçmek için hücre başına yaklaşık **1.800 enjeksiyon** gerekir.
- **Gerçekçilik sorunları:**
  - Yalnız tek-bit hata kullanılırsa SECDED her durumda %100 düzeltir ve ECC karşılaştırması anlamsızlaşır.
  - Scale ve zero-point metadata'sındaki (FP16) tek bir hata bütün grubu bozar.
  - Fiziksel BER modelinde ECC'li konfigürasyon daha fazla bit taşır; check bitleri de hata alabilir.
- **Önlem:**
  - Hiyerarşik örnekleme: önce layer × bit × K/V marjinalleri, sonra yalnız hassas hücrelerde head × token-tipi.
  - Wilson güven aralığı.
  - Fault planı **bir kez üretilir ve bir dosyaya yazılır**; bütün ECC konfigürasyonları aynı planı tekrar oynatır (common random numbers).
  - İki mod kullanılır: *data-bit paired* (mekanizma karşılaştırması) ve *physical-BER* (check bitleri de açıkta).
  - Metadata ayrı bir site sınıfı olarak modellenir.
  - M3'te multi-bit ve burst hata modelleri zorunludur.
- **Tehdit ettiği iddia:** SDC ve ECC sonuçları, heatmap'lerin istatistiksel geçerliliği.

### R4: Adil bütçe karşılaştırması ve bit hesabı

- **Risk:** Karma yapının metadata maliyeti (bit-width haritası, ECC mod haritası, tutulan indeksler, grup scale'leri) sayılmazsa adaptif sistem kâğıt üzerinde kazanır. Threshold taraması her metodu aynı bütçeye denk getirmez.
- **Önlem:**
  - Tek doğruluk kaynağı olarak bit hesabı modülü M0'da yazılır. Tahsis politikası da raporlama da bu modülü kullanır.
  - Eşit bütçe karşılaştırmaları için **bütçe kısıtlı bir allocator** (bit başına marjinal faydaya göre greedy veya knapsack) kullanılır. Threshold taraması yalnız Pareto noktaları için kullanılır.
  - Kabul toleransı: toplam bit farkı ±%1 (metadata dahil).
- **Tehdit ettiği iddia:** H2 (eşit bütçede adaptif > uniform), yani ana iddia.

### R5: Sinyal geçerliliği, confound'lar ve çalışma anında hesaplanabilirlik

- **(a) Büyüklük confound'u:** Quantization hassasiyeti ve fault hassasiyeti, ikisi de KV büyüklüğünün (outlier olma durumunun) bir göstergesi olabilir. O zaman yüksek korelasyon sahte olur ve politika "büyük değerleri koru" kuralına indirgenir.
- **(b) Çalışma anında bilinmezlik:** Token düzeyindeki hassasiyet inference sırasında bilinmez. Offline profil yalnızca (l, h, b, token-tipi) için önsel bilgi verebilir. Profil ile değerlendirme aynı veriden yapılırsa sızıntı olur.
- **(c) Pozisyon confound'u:** Semantik önem ile pozisyon (attention sink'leri, recency) birbirine karışır.
- **(d) Taban etkisi:** 1–2B modelde ajan başarısı düşük kalabilir; bu durumda etkiler görünmez.
- **Önlem:**
  - **Magnitude-only baseline** ("en büyük değerleri koru") zorunlu bir karşılaştırma olarak eklenir.
  - Kısmi korelasyon kullanılır.
  - Kalibrasyon ve test ayrımı yapılır.
  - Skor yalnızca çalışma anında hesaplanabilen sinyallerden kurulur; offline profil yalnızca (l, h, b) önseli olarak girer.
  - Sentetik probe'da pozisyonlar kontrollü yerleştirilir.
  - M6'da ikinci ve daha büyük bir model (Qwen3-4B) kullanılır.
- **Tehdit ettiği iddia:** H3, Q48, Q49.

### Ek riskler (daha küçük)

- M5'ten önce literatürde çok yakın bir çalışma çıkması (bkz. B.5).
- Windows'a özgü sorunlar: HF cache'i symlink kullanamıyor, bu yüzden daha fazla disk harcıyor; multiprocessing `spawn` kullanıyor. Tek GPU olduğundan paralellik batching ile sağlanır, GPU işleri için multiprocessing kullanılmaz.
- Qwen3'te QK-Norm var. Bu yüzden key outlier profili Llama ailesinden farklı olabilir ve literatürdeki KIVI gözlemleri birebir tekrar etmeyebilir. İkinci model ailesinin gerekçesi budur.

---

## D. Model seçimi

| Model | Layer | Q / KV head | d_head | KV/token (BF16) | Erişim | Not |
|---|---|---|---|---|---|---|
| **Qwen3-1.7B** | 28 | 16 / 8 | 128 | 112 KiB | Açık (Apache-2.0) | Native tool-call; non-thinking modu doğrulandı |
| Qwen3-4B | 36 | 32 / 8 | 128 | 144 KiB | Açık | Ölçek basamağı |
| Qwen2.5-1.5B-Instruct | 28 | 12 / **2** | 128 | 28 KiB | Açık | Yalnız 2 KV head, yani head ekseni fiilen yok |
| Llama-3.2-1B-Instruct | 16 | 32 / 8 | 64 | 32 KiB | Gated; **erişim açıldı** (2026-09-17) | rev `92131767…` |
| Llama-3.2-3B-Instruct | 28 | 24 / 8 | 128 | 112 KiB | Gated; **erişim açıldı** (2026-09-17) | rev `0cb88a4f…`; **KV geometrisi Qwen3-1.7B ile birebir aynı** |
| Gemma-3-1B | – | – | – | – | – | Yerel/global karışık attention; cache semantiği ilk faz için confound yaratır |

Qwen modellerinin sayıları HF config'lerinden okundu. Qwen3-1.7B için doğrulanan revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.

**Öneri: Qwen3-1.7B (birincil)**, revision sabitlenmiş, BF16, `sdpa`, non-thinking modunda.

1. **8 KV head:** Allocation'ın head ekseni KV-head'dir, query-head değil. Qwen2.5-1.5B'de bu eksen yalnızca 2 değer alır.
2. **Bütün katmanlar full attention:** Sliding window yok, dolayısıyla cache semantiği her katmanda aynı.
3. **Native tool-calling ve non-thinking modu doğrulandı:** M6'da model değiştirmek gerekmez. Model değişirse bütün hassasiyet profillerinin yeniden koşulması gerekir.
4. **Açık erişim:** Lisans uygun, revision sabitlenebilir, token gerekmez.
5. **Modern mimari:** d_head=128 ve 2:1 GQA, Llama-3 ve Qwen3-8B ile aynı sınıftadır. Bu, sonuçların büyük modellere taşınabileceği argümanını destekler.

**Bellek:** 4K token = 448 MiB, 16K = 1.75 GiB, 32K = 3.5 GiB KV. Ağırlıklar yaklaşık 3.5–4 GB (M0'da ölçülecek). 16 GB kartta 16K civarı context rahat sığar.

**Genelleme planı:**
- İkinci model ailesi: Llama-3.2-1B/3B-Instruct (erişim 2026-09-17'de açıldı). **Llama-3.2-3B, Qwen3-1.7B ile aynı KV geometrisine sahip** (28 × 8 × 128). Bu çift, cache geometrisi sabitken model ailesi (QK-Norm'lu ve QK-Norm'suz) etkisini ayırmak için kontrollü bir karşılaştırma sağlar; H2 ve Q48'in aileler arası tekrarı için birincil aday budur.
- Ölçek adımı: Qwen3-4B.
- 8B sınıfı modeller BF16'da 16 GB'a sığmaz. Ağırlık quantization'ı kullanmak confound yaratır; bu yüzden kapsam dışında ya da açık notla ekte kalır.

**Bilinçli maliyet:** Non-thinking modunda GSM8K doğruluğu thinking moduna göre düşüktür. Bu kabul edilebilir, çünkü ölçülen şey golden'a göre Δ'dır.

---

## E. İlk minimal dataset ve benchmark

| Dataset | Seviye | Ölçtüğü şey | M0 boyutu | Neden |
|---|---|---|---|---|
| **RAQ-Probe v0** (kendi üretimimiz, sentetik, seed'li) | L1 / L2 / L3 öncesi | Tip-etiketli bilgi geri çağırma ve tool-argument çıkarma: EM, cevap NLL'i, KL | 6 tip × 3 derinlik × 3 uzunluk (512/1K/2K) × 10 = 540 örnek | Token tipi ve pozisyon **yapısı gereği biliniyor**. Q49 bu etiketleri gerektiriyor; gerçek dataset'lerde bu etiket yok |
| **WikiText-2 (test)** | L1 | Cache-koşullu PPL ve KL (1536 token prefix cache'te, 512 token continuation skorlanır) | 40 parça | KV quantization literatürüyle trend karşılaştırması |
| **GSM8K (test)** | L1 reasoning | EM (greedy, non-thinking) | 200 | Çok adımlı reasoning; uzun decode'a duyarlılık |

**RAQ-Probe token tipleri:**
- `NUMBER`: tutar, miktar
- `ID`: ör. `INV-48213`
- `DATE`
- `ENTITY`: kişi, şirket
- `CONSTRAINT`: ör. "300 EUR'yu geçme"
- `TOOL_ARG`: context'teki bir değerin `<tool_call>` argümanına taşınması

Bilgiler context'in başına, ortasına ve sonuna yerleştirilir; bilgi ile filler'ın yeri değiştirilerek pozisyon confound'u kontrol edilir. `TOOL_ARG` tipi, BFCL'den önce ajan düzeyinde erken bir sinyal verir.

**Dürüstlük notu:** Cache-koşullu PPL, KIVI ve KVQuant'ın raporladığı standart PPL ile sayı sayı karşılaştırılamaz. Yalnızca trend karşılaştırılır.

**Sonraki aşamalar:**
- M4: RULER (NIAH single/multikey/multivalue, variable tracking; 4K/8K/16K), LongBench alt kümeleri (tek ve çok dokümanlı QA).
- M6: BFCL AST kategorileri (simple/multiple/parallel), deterministik çok turlu durum taşıma görevleri.
- tau-bench benzeri görevler bir user simulator LLM gerektirir (maliyet ve determinizm sorunu); M6'da ayrı bir karar noktasıdır.

---

## F. Repo mimarisi (son hali)

```
RAQ-KV/
├── README.md  pyproject.toml  requirements.txt  .gitignore
├── docs/
│   ├── 00_research_plan.md      # bu doküman
│   ├── claims.md                # iddia ↔ deney ↔ figür matrisi + önceden sabitlenmiş hipotez eşikleri
│   ├── conventions.md           # bit indeksleme, tensör yerleşimi, adres uzayı, birimler
│   └── decisions.md             # teknik karar günlüğü (ADR)
├── configs/
│   ├── models/  datasets/       # M0
│   ├── quant/  ecc/  faults/  pruning/  policies/    # M1+ bileşen ön ayarları
│   └── experiments/             # m0_golden.yaml, m1_kvquant.yaml, ... (sweep tanımları)
├── src/raq_kv/
│   ├── config/        # pydantic şeması, YAML yükleme, override, sweep açma
│   ├── models/        # loader.py, arch.py (KV geometrisi: L, H_q, H_kv, d_head, GQA grubu)
│   ├── kv/            # snapshot.py, address.py (l,h,t,c,K|V,bit ↔ düz indeks), layers.py (M5)
│   ├── budget/        # bit_accounting.py: tek doğruluk kaynağı
│   ├── quantization/  # base.py, uniform.py (2/4/8 bit, granülerlik), packing.py, stats.py
│   ├── ecc/           # base.py, none.py, parity.py, hamming_secded.py, vectors.py
│   ├── faults/        # bit_utils.py, models.py, plan.py (üret/kaydet/tekrar oynat), injector.py
│   ├── pipeline/      # kv_protection.py: Quantize→Encode→Inject→Decode→Dequantize (HW referans modeli)
│   ├── importance/    # random, recency, attention (gözlem penceresi + GQA toplama), frequency, perturbation, semantic_heuristic
│   ├── profiling/     # quant_sensitivity.py, fault_sensitivity.py, profile_store.py
│   ├── compression/   # base, full, random, attention, importance (RoPE pozisyon yönetimiyle)
│   ├── allocation/    # decisions.py, score.py (mult/add/learned), policies/{uniform,threshold,budgeted}.py
│   ├── data/          # base.py, raq_probe.py, wikitext.py, gsm8k.py, (M4) ruler.py, (M6) bfcl.py
│   ├── evaluation/    # protocol.py, metrics.py, reliability.py (SDC sınıflayıcı), memory.py, latency.py, toolcall.py
│   ├── experiments/   # runner.py, ids.py, provenance.py
│   ├── analysis/      # aggregate.py, stats.py (CI, paired bootstrap, McNemar), pareto.py, correlation.py, plots/
│   └── utils/         # io.py (append-only JSONL), seed.py, paths.py
├── scripts/           # run_experiment.py, aggregate.py, check_env.py, probe_cache_api.py, export_vectors.py
├── tests/             # CPU'da rastgele ağırlıklı küçük modelle unit testler; `gpu` işaretli entegrasyon testleri
├── outputs/           # git'e girmez: raw/<exp_id>/, csv/, figures/, reports/, cache/golden/, profiles/, fault_plans/
└── hardware/          # yalnız README; RTL, Python referansı dondurulduktan sonra (M3 sonrası)
```

**Önerilen yapıdan sapmalar ve gerekçeleri:**

1. **`src/raq_kv/` paket düzeni.** Üst seviyede `models` ve `utils` gibi isimler başka paketlerle çakışır. Bu düzenle `pip install -e .` ile kurulabilir.
2. **`int8.py`, `int4.py`, `int2.py` yerine tek bir parametrik `uniform.py`.** Algoritma aynı; bit genişliği config'ten gelir. Ayrı dosyalar kod tekrarına ve zamanla birbirinden sapmaya yol açar.
3. **Yeni `kv/address.py`.** Bit adresleme kuralını quantization, ECC, fault injection ve RTL test vektörleri ortak kullanır. Bu yüzden tek bir yerde tanımlanmalı.
4. **Yeni `budget/`.** Eşit bütçe karşılaştırmaları (§50–51) ancak allocator ile raporlama aynı formülü kullanırsa geçerli olur (R4).
5. **Yeni `pipeline/kv_protection.py`.** §37'deki donanım zinciri, birleştirilebilir tek bir nesne olarak yazılır. RTL için golden model bu olacak.
6. **`profiling/` ile `importance/` ayrı.** Profiling pahalı, offline koşan ve çıktısı cache'lenen kampanyalardır; importance ise çalışma anında hesaplanabilen skorlayıcılardır. Bu ayrım, R5'teki çalışma anında hesaplanabilirlik kısıtını koda yansıtır.
7. **Numaralı `experiments/0x_*/` kod klasörleri yerine `configs/experiments/*.yaml` ve tek bir runner.** Deneyler kod değil veridir; kopyala-yapıştır script'lerin önüne geçilir.
8. **`analysis/`, `evaluation/`'dan ayrı.** `evaluation/` inference başına metrik üretir; `analysis/` istatistik, Pareto ve grafikleri yapar.
9. **`docs/claims.md`.** "Bu deney hangi iddiayı kanıtlıyor?" sorusunun kalıcı cevabı burada tutulur.
10. **`hardware/` ertelendi.** Python referansı dondurulana kadar yalnız README içerir.

**Config:** pydantic v2 (`extra="forbid"`) + PyYAML. Hydra kullanılmıyor: daha ağır, ayrıca kendi çıktı dizini mantığı bizim devam ettirilebilir dosya düzenimizle çakışır. Pydantic yazım hatalarını yakalar; örneğin yanlış yazılmış bir `ber` alanı sessizce varsayılana düşüp bütün bir sweep'i bozamaz.

**Deney ID'si:** `exp_id = <okunabilir-slug>-<hash8>`, ör. `qwen3-1p7b_raqprobe_kv4_secded_single_p25_s42-3fa1c2e9`.
- Hash, çözümlenmiş config'in kanonik JSON'undan ve `schema_version` değerinden hesaplanır.
- Birim ID'si (exp_id, sample_id, fault_trial_id) üçlüsünden oluşur.
- Runner, `results.jsonl` içinde tamamlanmış birimleri tarar ve onları atlar.
- Aynı hash ile yeniden çalıştırılırsa kaldığı yerden devam eder.
- Git hash değişmişse uyarı verir ve `--allow-code-change` bayrağı ister.

**Ham veri:** `outputs/raw/<exp_id>/manifest.json` (config + provenance), `results.jsonl` (yalnızca eklenir, periyodik fsync), `errors.jsonl`. CSV her zaman ham veriden yeniden üretilir; ham veri hiçbir zaman silinmez veya üzerine yazılmaz.

**Golden çıktılar:** Tam vocab logits diske yazılmaz (151k × pozisyon, çok büyük). Aynı prefill snapshot'ından örnek başına bir kez hesaplanıp bellekte tutulur. Diske yalnızca özet golden kayıt yazılır: üretilen token'lar, token başına NLL, top-k logprob.

---

## G. Milestone planı (M0–M6 ve karar kapıları)

Her milestone için verilenler: **Amaç**, **Çıktılar**, **Çıkış kriteri**, **Desteklediği iddia**. Bir milestone'un çıkış kriterleri sağlanmadan sonrakine geçilmez.

### M0: Golden baseline ve ölçüm altyapısı

- **Amaç:** Bütün Δ ölçümlerinin referans noktasını kurmak ve ölçüm protokolünü doğrulamak.
- **İki ara kontrol noktası.** Protokol, 40 dosya ona bağımlı hale gelmeden önce doğrulanır:
  - **M0a:** model yükleyici, iki fazlı protokol ve kendi greedy döngümüz, metrikler, JSONL; yalnızca WikiText-2. İlk gerçek ölçüm burada alınır.
  - **M0b:** config şeması, resume, provenance, RAQ-Probe, GSM8K, test paketinin tamamı.
- **Çıktılar:**
  - git deposu ve sabit sürümlü ortam
  - config, ID, provenance, resume ve JSONL altyapısı
  - Qwen3-1.7B yükleyici
  - RAQ-Probe v0, WikiText-2 ve GSM8K-200
  - İki fazlı protokol
  - Metrikler: token sayısı, NLL/PPL, EM, TTFT, TPOT, e2e, tokens/s, KV byte (gerçek ve analitik), tepe bellek
  - `bit_accounting` v0 (FP16 yolu)
- **Çıkış kriterleri:**
  1. Gerçek modelde tek geçiş ile iki fazlı yol aynı argmax'ı verir. KL değeri kaydedilir; float32'de KL < 1e-4 şartı aranır. BF16'da kernel yolları farklı olabileceği için KL yalnızca raporlanır, geçme koşulu değildir. Golden **tanım gereği** iki fazlı yoldur ve eşleştirme aynı snapshot üzerinden yapılır; bu kontrol bir sağlamadır. İzleme: `docs/m0_exit.md`.
  2. Aynı seed ile yeniden koşulduğunda token'lar ve metrikler aynıdır. Deterministik olmayan kernel'ler varsa kaydedilir.
  3. Analitik KV byte = tensörlerin nbytes toplamı (tam eşitlik). Ayrıca 2K token'da `memory_allocated` farkıyla en fazla %2 sapma.
  4. Runner deneyin ortasında öldürülüp yeniden başlatıldığında tekrarlanan birim olmaz.
  5. Her kayıtta provenance bulunur; CSV ham veriden yeniden üretilebilir.
  6. Decode forward/s ölçülür (batch kuralına uygun olarak: aynı örneğin varyantları). M2 kampanya bütçesi bu değerden hesaplanır.
  7. **RAQ-Probe golden doğruluk bandı:** Golden EM her tip ve her derinlik için %85–100 aralığında olmalıdır. Altında kalan hücrelerde fault ölçümleri taban etkisi yüzünden gürültüye dönüşür ve Q49 cevaplanamaz. Bu durumda zorluk ayarlanıp probe yeniden üretilir. Üst sınırda tavan etkisi de izlenir: %100'e yapışan tipte küçük hasarlar da cevabı değiştirmeyebilir, bu yüzden EM'nin yanında cevap NLL'i ve KL de raporlanır.
- **Desteklediği iddia:** Deney düzeninin güvenilirliği; bütün Δ'ların sıfır noktası.

### M1: KV quantization

- **Amaç:** Precision azaldıkça kalitenin nasıl değiştiğini ölçmek ve quantization hassasiyeti haritasını çıkarmak.
- **Çıktılar:**
  - `kv/address.py` ve `docs/conventions.md` kesinleşir
  - Uniform quantizer (simetrik/asimetrik; per-tensor, per-channel, per-token, grup)
  - Bit packing
  - Quantization hata istatistikleri (MSE, SQNR, maksimum hata)
- **Taranan konfigürasyonlar:** BF16 passthrough, INT8, INT4, INT2 × granülerlik. KIVI tarzı key per-channel / value per-token referans alınır.
- **Quantization hassasiyet profili:** Tek bir (l, h, K/V, token-tipi) bölgesi düşük bite indirilir, geri kalan FP bırakılır. ΔKL ve ΔEM ölçülür.
- **Çıkış kriterleri:**
  - Passthrough sonucu M0 golden ile bit düzeyinde aynıdır.
  - INT8 golden'a çok yakındır (KL ≈ 0).
  - Bit azaldıkça kalite monoton düşer.
  - Literatürle nitel uyum vardır (ör. 2 bitte key için per-channel, per-token'dan iyi olmalı).
  - Etkin bit genişliği metadata dahil raporlanır.
- **Desteklediği iddia:** Quantization hassasiyetinin heterojen olduğu; Q48'in girdisi.

### M2: Fault injection (ECC yok)

- **Amaç:** Fault kırılganlığı haritası; H1'in ilk testi.
- **Çıktılar:**
  - `bit_utils` (bit sırası: LSB = bit 0)
  - Fault modelleri: single, MSB, LSB şimdi; double, rastgele BER ve burst için arayüz hazır
  - Fault planı dosyası
  - Injector: tamsayı kodlarda, BF16 baseline'da ve metadata'da
- **Hiyerarşik kampanya:** (A) layer × bit × K/V marjinalleri; (B) hassas hücrelerde head × token-tipi.
- **Sonuç sınıfları:** masked / token düzeyinde değişiklik / görev düzeyinde SDC / NaN-Inf.
- **Çıkış kriterleri:**
  - Injector'ın yalnızca istenen biti çevirdiği unit testle kanıtlanır.
  - Heatmap'ler Wilson CI ile raporlanır.
  - Aynı fault planı iki kez koşulduğunda aynı sonuç çıkar.
- **KAPI-1:** Corr(QS, FS) ham ve büyüklük kontrollü kısmi korelasyon olarak hesaplanır; Corr(Semantic, FaultImpact) pilot olarak bakılır. Bit pozisyonu sabitken heterojenlik yoksa (bkz. I, H1'in red koşulu) tez yeniden değerlendirilir. Korelasyonun yönü M5'te skorun tasarımını belirler: tek ortak sinyal mi, çok sinyalli bir yapı mı?
- **Desteklediği iddia:** C2 (karakterizasyon), H1, Q48, Q49 (pilot).

### M3: ECC

- **Amaç:** Koruma seviyelerinin maliyet ve fayda karşılaştırması; RTL için golden vektörler.
- **Çıktılar:**
  - `none`, `parity` (kelime başına çift parity), `hamming_secded` (genişletilmiş Hamming; (13,8), (22,16), (39,32), (72,64) yapılandırılabilir)
  - Metadata koruması
  - Golden vektör dışa aktarımı (CSV/HEX/NPY, §39'daki alanlarla)
  - DUE politikası config ile seçilir: maskele (budanmış say), sıfırla veya yeniden hesapla (maliyeti hesaba katılarak)
- **Unit testler:**
  - Küçük kodlarda **bütün** 1-bit ve 2-bit hata desenleri (ör. (13,8) için tamamı) denenir.
  - Hatasız veri → orijinal veri; 1 bit → düzeltilir; 2 bit → tespit edilir, düzeltilemez; parity davranışı doğru.
- **Eşleştirilmiş kampanya:** Aynı fault planı NONE, PARITY ve SECDED üzerinde tekrar oynatılır. Hem data-bit paired hem physical-BER modu. Double, rastgele BER ve burst hata modelleri eklenir.
- **Metrikler:** Tespit oranı, düzeltme oranı, SDC, DUE, overhead, McNemar testi.
- **KAPI-2 (oracle headroom pilotu):** M2 profili kullanılarak (oracle) eşit bütçede karma tahsis yapılır ve Uniform INT4 + Uniform SECDED ile karşılaştırılır. **Oracle bile uniform'u geçemiyorsa H2 erken reddedilir** ve makale karakterizasyon ile ajan SDC eksenine döner.
- **Desteklediği iddia:** Güvenilirlik/overhead dengesi; uniform ECC baseline'ının gücü; B.3 alt hipotezi.

### M4: Token sıkıştırma

- **Amaç:** Hangi içerik tiplerinin tutulması gerektiğini ve semantik önemin attention'a göre katkısını ölçmek.
- **Çıktılar:**
  - Önem skorlayıcıları:
    - random, recency
    - attention (SnapKV tarzı gözlem penceresi; GQA toplama yöntemi, ortalama ya da maksimum, belgelenir)
    - frequency/redundancy
    - perturbation (küçük alt kümede leave-one-out)
    - semantic_heuristic (sayı, ID, tarih, varlık, kısıt anahtar kelimeleri, tool JSON'u ve system bölgesi için tipleyici)
  - Doğru RoPE pozisyonlarıyla budama; oranlar 0, 10, 25, 50, 75%
  - H2O ve StreamingLLM tarzı yeniden uygulamalar ("reimplementation" etiketiyle)
  - Level 2 başlangıcı: RULER alt kümesi
- **Çıkış kriteri:** %0 budama golden ile bit düzeyinde aynıdır. Tutulan token oranı her kayıtta bulunur.
- **Desteklediği iddia:** C3'ün retention bileşeni; Q49.

### M5: Ortak güvenilirlik tahsisi (ana katkı)

- **Amaç:** (r, b, e) kararlarını tek bir pipeline'da vermek.
- **Çıktılar:**
  - Skorlar: çarpımsal, toplamsal (ağırlık taramasıyla) ve öğrenilmiş (kalibrasyon ayrımıyla lojistik veya ridge regresyon; **RL yok**)
  - Politikalar: threshold (Pareto için), **budgeted** (eşit bütçe için birincil)
  - Bit-plane koruması: ör. INT4'ün yalnız üst 2 bitine SECDED. R(l,h,t,b) içindeki b ekseni bu şekilde operasyonel hale gelir.
  - Decode sırasında üretilen yeni KV için Cache/Layer subclass
- **Baseline'lar:**
  - B0–B6
  - Uniform ECC ve Adaptive ECC
  - **Magnitude-only**
  - KIVI tarzı, H2O/SnapKV tarzı
- **Ablation'lar:** §19'daki tamamı.
- **Eşit bütçe kuralları:** Toplam bit farkı ±%1, aynı fault planı, aynı seed'ler, aynı tutulan token bütçesi.
- **Çıkış kriterleri:**
  - Eşleştirilmiş istatistik: 5 seed, paired bootstrap, Holm düzeltmesi.
  - Pareto grafikleri ham veriden üretilir.
- **Desteklediği iddia:** H2, H3, C1 ve C3.

### M6: Ajan benchmark'ı ve genelleme

- **Amaç:** Bit hatalarının ve sıkıştırmanın gerçek tool çağrısına ve çok adımlı görevlere etkisini göstermek.
- **Çıktılar:**
  - Tool-call değerlendirme: JSON ayrıştırma, fonksiyon adı doğruluğu, tipe duyarlı argüman eşleşmesi (sayılar), görev başarısı
  - BFCL AST alt kümeleri
  - Deterministik çok turlu durum taşıma görevleri
  - Pass@1 (greedy)
  - **Pass^k**: bağımsız k fault gerçekleşmesinin hepsinde başarı (güvenilirlik odaklı tanım)
  - Metrik uyuşmazlığı analizi: KL hasarı ile görev başarısızlığı arasındaki sıra korelasyonu
- **Genelleme:** Qwen3-4B ve (erişim açılırsa) Llama-3.2.
- **Karar noktası:** tau-bench tarzı user simulator (maliyet ve determinizm).
- **Desteklediği iddia:** C4 (ajan düzeyinde SDC).

### Paralel iş kolu

- M0–M4 boyunca sistematik literatür taraması.
- `docs/claims.md` her kapıda güncellenir.
- Hipotez eşikleri **M2 başlamadan önce** donmuş olur.

---

## H. M0'da oluşturulacak dosyalar

**Kök ve belgeler**

| # | Dosya | Amaç |
|---|---|---|
| 1 | `.gitignore` | outputs/, .venv/, HF cache, `__pycache__` |
| 2 | `pyproject.toml` | `raq_kv` paketi (src düzeni), Python ≥3.11, pytest işaretleri (`gpu`, `slow`), ruff ayarları |
| 3 | `requirements.txt` | Sabit sürümler: torch 2.11.0 (cu128), transformers 5.17.0, datasets, pydantic, pyyaml, numpy, pandas, matplotlib, pytest |
| 4 | `README.md` | Kurulum (Windows ve Linux), M0'ın çalıştırılması, çıktı düzeni |
| 5 | `docs/00_research_plan.md` | Bu doküman |
| 6 | `docs/claims.md` | İddia ↔ deney ↔ figür matrisi; önceden sabitlenmiş eşikler |
| 7 | `docs/conventions.md` | `[B, H_kv, T, d_head]` yerleşimi; adres üçlüleri; bit sırası (LSB = 0); token indeksi = orijinal pozisyon; `position_ids` (RoPE, orijinal pozisyon, her zaman açık) ile cache'in fiziksel uzunluğu (maske ve sorgu offset'i, transformers tarafından otomatik) ayrımı; değiştirilmiş cache'te `generate()` yasağı; batch kuralı; birimler (bit ve byte, KiB) |
| 8 | `docs/decisions.md` | ADR günlüğü: model, transformers sürüm kilidi, iki fazlı protokol, simüle edilmiş quantization |

**Config dosyaları**

| # | Dosya | Amaç |
|---|---|---|
| 9 | `configs/models/qwen3_1p7b.yaml` | İsim, revision SHA, dtype, attention implementasyonu, chat template argümanları (`enable_thinking: false`) |
| 10 | `configs/datasets/raq_probe_v0.yaml` | Tipler, derinlikler, uzunluklar, örnek sayısı, generator seed'i |
| 11 | `configs/datasets/wikitext2_ppl.yaml` | Parça sayısı, prefix ve continuation uzunlukları, dataset revision |
| 12 | `configs/datasets/gsm8k_200.yaml` | Alt küme seçimi (seed), cevap çıkarma, maksimum yeni token |
| 13 | `configs/experiments/m0_golden.yaml` | Model + 3 dataset + seed'ler [0, 1, 2] + latency tekrar sayısı |

**Paket kodu (`src/raq_kv/`)**

| # | Dosya | Amaç |
|---|---|---|
| 14 | `__init__.py` | Sürüm |
| 15 | `config/schema.py` | Pydantic modelleri (`extra="forbid"`), `schema_version` |
| 16 | `config/loader.py` | YAML okuma, include'lar, noktalı CLI override, grid sweep açma |
| 17 | `models/loader.py` | Model ve tokenizer'ı config'e göre yükleme (revision, dtype, attention); parametre byte sayısı |
| 18 | `models/arch.py` | `KVGeometry`: L, H_q, H_kv, d_head, GQA grup boyu, dtype (config'ten okunur, sabit kodlanmaz) |
| 19 | `kv/snapshot.py` | `DynamicCache` → `KVSnapshot`: çıkarma, klonlama, geri yükleme; transformers sürüm koruması |
| 20 | `budget/bit_accounting.py` | `BitBudget`. M0'da FP16 yolu hazır. Veri, metadata, ECC ve indeks haritası terimleri iskelet halinde; henüz yazılmamış kısımlar sessizce 0 döndürmez, `NotImplementedError` fırlatır |
| 21 | `data/base.py` | `Sample` (id, context, query, target, tip-etiketli span'ler), dataset kayıt defteri |
| 22 | `data/raq_probe.py` | Seed'li generator: filler + tipli bilgiler, kontrollü derinlik, karakter→token span eşlemesi |
| 23 | `data/wikitext.py` | WikiText-2 test parçaları, token düzeyinde prefix/continuation bölmesi |
| 24 | `data/gsm8k.py` | GSM8K alt kümesi, sayısal cevap çıkarma |
| 25 | `evaluation/protocol.py` | İki fazlı protokol: prefill → snapshot → (M0'da kimlik) dönüşüm → teacher-forced skor veya açık `position_ids` ile greedy üretim |
| 26 | `evaluation/metrics.py` | NLL, PPL, EM (normalize edilmiş), token logprob'ları, KL (özdeş dağılımlarda 0 testiyle) |
| 27 | `evaluation/latency.py` | TTFT, TPOT, e2e, tokens/s; `cuda.synchronize`, ısınma turu, tekrar |
| 28 | `evaluation/memory.py` | Model byte'ları; KV byte'ları (gerçek ve analitik); allocated/reserved farkları ve tepe değerler |
| 29 | `experiments/ids.py` | Kanonik JSON, config hash'i, slug, birim ID'leri |
| 30 | `experiments/provenance.py` | Python, torch, transformers, CUDA, cuDNN, GPU, sürücü, model revision, git hash (`-dirty` eki), zaman damgası |
| 31 | `experiments/runner.py` | Manifest oluşturma ve doğrulama, tamamlanmış birimleri atlama, hata yakalama, üzerine yazmama |
| 32 | `utils/io.py` | Yalnızca eklenen JSONL (flush + fsync); atomik JSON yazma; yarım kalmış son satırı tolere eden okuyucu |
| 33 | `utils/seed.py` | Bütün RNG'lerin seed'lenmesi, deterministik bayraklar, `CUBLAS_WORKSPACE_CONFIG` |
| 34 | `utils/paths.py` | Repo kökünü bulma (pathlib), çıktı dizinleri |
| 35 | `analysis/aggregate.py` | Ham JSONL → özet CSV (seed'ler üzerinden ortalama, std, n, %95 CI) |

**Script'ler**

| # | Dosya | Amaç |
|---|---|---|
| 36 | `scripts/check_env.py` | Ortamı doğrular ve `outputs/reports/env.json` dosyasına yazar |
| 37 | `scripts/probe_cache_api.py` | §0'daki fizibilite testinin kalıcı hali; her transformers yükseltmesinde tekrar koşulur |
| 38 | `scripts/run_experiment.py` | CLI: `--config`, `--set a.b=c`, `--resume` |
| 39 | `scripts/aggregate.py` | Özetleme CLI'ı |

**Testler**

| # | Dosya | Amaç |
|---|---|---|
| 40 | `tests/conftest.py` | Rastgele ağırlıklı küçük Qwen3 fixture'ı (CPU, indirme yok); `gpu` işaretli testleri atlama |
| 41 | `tests/test_protocol_equivalence.py` | (i) Tek geçiş = iki fazlı yol. (ii) Kendi greedy döngümüz = `generate()` (değiştirilmemiş cache'te). (iii) **Budama eşdeğerliği:** ortadan budanmış cache ile doğru `position_ids` kullanılarak yapılan çok token'lı decode, budanan pozisyonları 2D `attention_mask` ile gizlenmiş tam cache ile yapılan decode ile aynı sonucu verir. (iv) Yanlış pozisyon verildiğinde bu eşdeğerlik bozulur; test, pozisyon hatasının yakalanabildiğini doğrular |
| 42 | `tests/test_kv_snapshot.py` | Snapshot gidiş-dönüşü bit düzeyinde aynı; mutasyon decode'u değiştiriyor |
| 43 | `tests/test_bit_accounting.py` | Analitik byte = tensör nbytes (farklı T ve GQA oranlarında) |
| 44 | `tests/test_ids_resume.py` | Aynı config → aynı ID; farklı config → farklı ID; kesilip devam edilen koşuda tekrarlanan birim yok |
| 45 | `tests/test_io.py` | Yarım son satır toleransı; atomik yazma |
| 46 | `tests/test_raq_probe.py` | Seed determinizmi; span etiketleri doğru token'lara eşleniyor; cevap context'te tam bir kez geçiyor |
| 47 | `tests/gpu/test_real_model_m0.py` | Qwen3-1.7B üzerinde iki fazlı eşdeğerlik ve bellek formülü ile `memory_allocated` karşılaştırması |

Ayrıca M0'ın ilk adımı olarak `git init` yapılır ve ilk commit oluşturulur, çünkü provenance her kayıtta commit hash'i ister. `outputs/` altındaki alt dizinler için `.gitkeep` dosyaları eklenir.

---

## I. Hipotezin desteklenmesi ve reddedilmesi

**Kural:** Aşağıdaki sayısal eşikler başlangıç önerisidir. **M2 başlamadan önce `docs/claims.md` içinde dondurulur** ve bütün sonuçlar raporlanır. Veriye bakıldıktan sonra eşik değiştirilmez.

### H1: Heterojenlik (temel önerme)

- **Ölçüm:** P(SDC | s konumunda hata) ve E[KL | s konumunda hata), **bit pozisyonu sabit tutularak** (MSB'nin LSB'den önemli olması zaten beklenir), layer, KV-head ve token-tipi eksenlerinde.
- **Destek:**
  - Her bit pozisyonu içinde en hassas %10'luk hücre, toplam hasarın ≥ %50'sini taşır.
  - Hücreler arası farklar CI ile ayrışır.
  - Sonuç ≥ 2 dataset ve ≥ 3 seed'de tekrarlanır.
- **Red:** Bit pozisyonu kontrol edildiğinde en hassas %10 hücre hasarın yalnızca %10–20'sini taşır. O zaman adaptif korumanın gerekçesi yoktur ve çalışma karakterizasyon makalesine döner.

### H2: Eşit bütçede adaptif > uniform (ana iddia)

- **Destek:**
  - Toplam bit farkı ±%1 (metadata dahil), aynı fault planı ve aynı seed'lerle, OURS, Uniform INT4 + Uniform SECDED'e göre görev düzeyinde SDC'yi ve/veya kalite kaybını anlamlı biçimde düşürür (paired bootstrap %95 CI 0'ı içermez; McNemar p < 0.05, Holm düzeltmeli).
  - Bu sonuç ≥ 2 model ailesinde ve Level 1 + Level 2 görevlerinde (tercihen Level 3'te de) görülür.
  - **OURS magnitude-only baseline'ı da geçer.**
- **Red (herhangi biri):**
  - Fark CI'nın içinde kalır.
  - Kazanç yalnızca metadata bitleri sayılmadığında ortaya çıkar.
  - Magnitude-only baseline OURS'a yetişir.
  - KAPI-2'de oracle bile uniform'u geçemez.

### H3: Ortak skor, tek sinyalden iyi

- **Destek:** Eşit bütçede tam skor, en iyi tek sinyalli politikadan anlamlı biçimde iyidir ve her ablation sonucu kötüleştirir.
- **Red:** Tek bir sinyal (attention importance veya büyüklük) aynı sonucu verir. Bu durumda katkı "mevcut KV sıkıştırma + ECC birleşimi"ne iner.

### Q48: Corr(QuantSens, FaultSens)

- **Ölçüm:** Spearman ρ, (l, h, K/V) ve (l, h, token-tipi) granülerliğinde, bootstrap CI ile. Ayrıca KV büyüklüğü kontrol edilerek **kısmi ρ**.
- **Kısmi ρ ≥ 0.6:** Tek ortak sinyal gerekçelidir; basit bir ortak skor yeterli olur.
- **Kısmi ρ ≤ 0.3:** Ayrı sinyaller gerekir; çok sinyalli bir allocator kurulur. **Makale açısından bu daha güçlü bir tez olur:** "Quantization'a bakarak yapılan koruma yetersizdir."
- **İkisinin arası:** Kısmi ilişki.
- **Ham ρ yüksek, kısmi ρ düşük:** Ortak neden büyüklüktür. Büyüklük açık bir sinyal olarak modele girer ve novelty iddiası daralır. Bu sonuç da raporlanır.

### Q49: Semantik önem ve fault etkisi

- **Destek:** Pozisyon, layer ve bit pozisyonu eşleştirildiğinde kritik tiplere (sayı, ID, tool argümanı, kısıt) enjekte edilen hataların görev düzeyinde SDC olasılığı filler'a göre yüksektir (odds ratio > 1, CI 1'i içermez). Ayrıca bu etki KL tabanlı metrikte görülenden büyüktür, yani token düzeyindeki metrik hasarı hafife alır.
- **Red:** Eşleştirilmiş koşulda OR ≈ 1 çıkar. Bu, fault etkisinin içerikten bağımsız olduğunu gösterir; semantik sinyal yalnızca retention için anlamlı olur, koruma için olmaz.

---

## J. Gerçek katkı ve mühendislik katkısı

### Bilimsel katkı adayları (makalenin iddiaları)

| # | Katkı | Kanıtlayan deneyler |
|---|---|---|
| C1 | **Ortak bit bütçesi formülasyonu:** retention, precision ve protection kararları tek para biriminde; metadata dahil eşit bütçe değerlendirme protokolü | M3 KAPI-2, M5 |
| C2 | **Quantized KV-cache için çok granülerlikte kırılganlık karakterizasyonu** (layer × KV-head × K/V × token-tipi × bit × metadata) ve quantization hassasiyeti ile semantik önemle ilişkisi, büyüklük confound'u kontrol edilmiş halde. **Sonucun yönünden bağımsız olarak katkıdır.** | M1, M2, KAPI-1 |
| C3 | **Çalışma anında hesaplanabilen ortak güvenilirlik skoru ve bütçe kısıtlı tahsis:** eşit bütçede uniform, tek sinyalli ve magnitude-only alternatiflere göre Pareto üstünlüğü | M5 (H2, H3) |
| C4 | **Ajan düzeyinde SDC:** PPL ve KL'nin tool argümanı ve durum hatalarını hafife aldığının gösterilmesi; Pass^k'nın fault gerçekleşmeleri üzerinden güvenilirlik metriği olarak tanımlanması | M6 (ön sinyal: RAQ-Probe `TOOL_ARG`, M2) |
| C5 (şartlı) | PBRT ve ReliabilityEfficiency metrikleri. Yalnızca mevcut metriklerin ayırt edemediği konfigürasyonları ayırt ettikleri gösterilirse katkı sayılır | M5 |
| C6 (şartlı, destekleyici) | Adaptifliğin donanım maliyeti: çok modlu ECC codec'i ve mod metadata'sının LUT/FF/Fmax maliyetinin kazanılan bitleri yemediğinin gösterilmesi | RTL aşaması |
| B.3 (alt bulgu) | Tespit edilen hatanın zorla eviction'a dönüşmesi: PARITY + önem-farkındalıklı retention | M3, M5 |

### Mühendislik katkısı (makalede "setup" veya "artifact" olarak geçer, iddia değildir)

- Transformers 5.x KV snapshot ve Cache katmanı; iki fazlı protokol
- Uniform KV quantizer'lar (KIVI/KVQuant tarzı yöntemlerin yeniden uygulanması)
- Parity ve Hamming SECDED (ders kitabı kodları)
- Fault injector (PyTorchFI benzeri araçların KV'ye uyarlanması)
- Budama baseline'ları (H2O, SnapKV, StreamingLLM tarzı yeniden uygulamalar)
- Runner, ID, resume ve provenance altyapısı; analiz ve grafik araçları
- Golden test vektörü dışa aktarımı ve RTL modülleri
- Açık kaynak artifact. Reproducibility rozetleri için değerlidir ama katkı iddiası değildir.

### Gri alan

**RAQ-Probe** tek başına bir katkı değildir. Ancak tip etiketli ve pozisyonu kontrol edilmiş tasarımı Q49 için zorunludur; bu yüzden "değerlendirme metodolojisi"nin bir parçası olarak sunulabilir.

---

## K. Kararlar

### Kullanıcıdan beklenenler (2 madde)

1. **Birincil model: Qwen3-1.7B** (revision `70d244cc…`). §43 gereği model seçimi kullanıcıya ait; gerekçe D bölümünde.
2. **Llama-3.2 erişim talebi.** HF hesabından yapılması gerekiyor; onay süresi dışarıya bağlı olduğu için şimdiden başlatılmalı. İkinci model ailesi için gerekli; M0'ı engellemiyor.

### Kendi başıma aldığım kararlar (itiraz yoksa bu şekilde uygulanacak)

- **Ortam:** Proje içinde temiz `.venv`, sabit sürümlü torch 2.11.0+cu128 (yaklaşık 3 GB indirme). Global Python'daki paketler deneylere karışmasın diye böyle; reproducibility için gerekli.
- **transformers 5.17.0** sürüm kilidi. Fizibilite testi bu sürümde yapıldı.
- **Config:** pydantic v2 + PyYAML. Küçük bağımlılıklar; yazım hatalarını yakalar.
- **M0 iki adımda:** M0a (WikiText ile protokol doğrulaması) ve M0b (tam altyapı).

### İleride birlikte verilecek karar

- **I bölümündeki sayısal eşikler** M2 başlamadan önce `docs/claims.md` içinde birlikte dondurulacak.
