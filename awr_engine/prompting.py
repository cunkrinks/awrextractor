def build_prompt(
    analysis_level="technical",
    recommendation_level="medium",
    language="id",
    style="hybrid",
):
    # ============================
    # 1. Bahasa
    # ============================
    if language == "id":
        lang_intro = (
            "Gunakan Bahasa Indonesia teknis yang jelas, ringkas, dan profesional. "
            "Istilah Oracle seperti wait event, latch, GC, dan parameter tetap gunakan Bahasa Inggris."
        )
    else:
        lang_intro = (
            "Use clear, concise, and professional technical English. "
            "Oracle terms such as wait events, latches, GC, and parameters must remain unchanged."
        )

    # ============================
    # 2. Level Analisis
    # ============================
    if analysis_level == "executive":
        analysis_block = """
Berikan analisis tingkat eksekutif yang ringkas dan mudah dipahami:
- Fokus pada gambaran besar performa sistem.
- Soroti tren utama CPU, AAS, wait events, dan I/O.
- Hindari detail teknis mendalam atau parameter Oracle.

Wajib sertakan:
1. Ringkasan Status
   - Tentukan status keseluruhan: Sehat / Peringatan / Kritis.
   - Dasarkan pada pola CPU, AAS, wait events, dan I/O.

2. Temuan Utama
   - Daftar 3–5 anomali paling signifikan.
   - Gunakan bahasa sederhana namun akurat.

3. Akar Masalah (Root Cause)
   - Jelaskan penyebab utama secara high-level.
   - Tunjukkan hubungan antar metrik tanpa detail teknis.

4. Analisis Tren
   - CPU Trend Analysis
   - Wait Event Analysis
   - AAS Trend Analysis
   - I/O Analysis
   - RAC Analysis (jika ada)

Gunakan bahasa ringkas, langsung ke inti masalah, dan fokus pada dampak bisnis."""

    elif analysis_level == "technical":
        analysis_block = """
Berikan analisis teknis lengkap dan terstruktur:
- Bahas CPU, wait events, AAS, I/O, RAC, dan SQL.
- Identifikasi bottleneck utama dan pola antar snapshot.
- Jelaskan hubungan antar metrik (CPU ↔ AAS ↔ wait events).

Wajib sertakan:
1. Ringkasan Status
   - Tentukan status: Sehat / Peringatan / Kritis.
   - Jelaskan alasan teknisnya.

2. Temuan Utama
   - Daftar 5–8 anomali atau pola tidak normal.
   - Sertakan korelasi antar metrik.

3. Akar Masalah (Root Cause)
   - Jelaskan penyebab utama berdasarkan data.
   - Tunjukkan hubungan sebab-akibat antar komponen.

4. Analisis Tren Detail
   - CPU Trend Analysis
   - Wait Event Analysis
   - AAS Trend Analysis
   - I/O Analysis
   - RAC Analysis
   - Top SQL Analysis

Gunakan bahasa teknis yang jelas dan profesional."""

    else:  # deepdive
        analysis_block = """
Berikan analisis tingkat ahli dengan kedalaman maksimal:
- Bahas parameter Oracle yang relevan (misal: _optimizer_use_feedback, parallel_degree_policy).
- Jelaskan perilaku optimizer, concurrency, GC, latch, dan I/O subsystem.
- Sertakan insight tingkat ahli seperti GC tuning, interconnect latency, dan plan stability.

Wajib sertakan:
1. Ringkasan Status
   - Tentukan status: Sehat / Peringatan / Kritis.
   - Sertakan indikator teknis yang mendasari.

2. Temuan Utama
   - Daftar 8–12 temuan teknis mendalam.
   - Sertakan korelasi multi-metrik (CPU ↔ AAS ↔ I/O ↔ GC).

3. Akar Masalah (Root Cause)
   - Jelaskan penyebab utama secara mendalam.
   - Bahas interaksi antar komponen internal Oracle.

4. Analisis Mendalam
   - CPU Trend Analysis
   - Wait Event Analysis
   - AAS Trend Analysis
   - I/O Analysis
   - RAC Analysis
   - Top SQL Analysis
   - Insight tambahan (jika relevan)

Gunakan bahasa teknis tingkat ahli, namun tetap terstruktur dan mudah diikuti."""

    # ============================
    # 3. Level Rekomendasi
    # ============================
    if recommendation_level == "high":
        rec_block = """
Berikan rekomendasi tingkat tinggi:
- Fokus pada area perbaikan umum.
- Hindari parameter teknis atau konfigurasi mendalam.

Wajib sertakan:
1. Rekomendasi Tindakan
   - 3–5 langkah perbaikan yang jelas dan actionable.
   - Fokus pada dampak bisnis dan stabilitas sistem.

2. Prioritas
   - Urutkan rekomendasi berdasarkan urgensi."""
    
    elif recommendation_level == "medium":
        rec_block = """
Berikan rekomendasi teknis tingkat menengah:
- Sertakan insight actionable yang dapat langsung dieksekusi DBA.
- Hindari parameter terlalu spesifik kecuali sangat relevan.

Wajib sertakan:
1. Rekomendasi Tindakan
   - 5–8 langkah perbaikan teknis.
   - Sertakan konteks mengapa rekomendasi tersebut penting.

2. Prioritas
   - Urutkan berdasarkan dampak dan kompleksitas."""
    
    else:  # expert
        rec_block = """
Berikan rekomendasi tuning tingkat ahli:
- Sertakan parameter Oracle yang relevan.
- Sertakan saran plan stability, parallelism, PGA/SGA tuning, dan I/O optimization.
- Berikan langkah konkret yang dapat dieksekusi DBA senior.

Wajib sertakan:
1. Rekomendasi Tindakan
   - 8–12 langkah tuning tingkat ahli.
   - Sertakan parameter, contoh konfigurasi, atau teknik tuning.

2. Prioritas
   - Urutkan berdasarkan dampak teknis dan risiko.

3. Catatan Teknis
   - Sertakan insight tambahan jika ada potensi side-effect."""

    # ============================
    # 4. Style (Hybrid)
    # ============================
    style_block = """
Gunakan gaya hybrid:
- Struktur laporan mengikuti AWR.
- Insight mendalam mengikuti ADDM.
- Tambahkan interpretasi AI untuk menjelaskan pola, anomali, dan korelasi.
"""

    # ============================
    # 5. Struktur laporan wajib
    # ============================
    structure_block = """
Struktur laporan wajib:

1. Executive Summary
2. Ringkasan Status
   - Tentukan status keseluruhan: Sehat / Peringatan / Kritis
   - Berdasarkan tren CPU, AAS, wait events, dan I/O
3. Temuan Utama
   - Daftar anomali atau pola tidak normal yang ditemukan
   - Fokus pada perubahan signifikan antar snapshot
4. CPU Trend Analysis
5. Wait Event Analysis
6. AAS Trend Analysis
7. I/O Analysis
8. RAC Analysis
9. Top SQL Analysis
10. Akar Masalah (Root Cause)
    - Jelaskan hubungan antar metrik
    - Identifikasi penyebab utama berdasarkan korelasi data
11. Rekomendasi Tindakan
    - Berikan langkah perbaikan yang konkret dan terurut
    - Sesuaikan dengan tingkat analisis dan rekomendasi
12. Actionable Recommendations (jika level expert)

Aturan penting:
- Jangan membuat angka baru yang tidak ada di konteks.
- Jika data tidak tersedia, tulis "data tidak tersedia".
- Fokus pada tren antar snapshot, bukan nilai tunggal.
- Jelaskan hubungan sebab-akibat (misal: CPU tinggi → AAS naik → wait event meningkat).
"""

    # ============================
    # FINAL PROMPT
    # ============================

    timestamp_rules = """
Saat menyebutkan SNAP_ID, selalu sertakan timestamp dalam format:
YYYY-MM-DD HH24:MI

Contoh:
"SNAP_ID 3450 (2024-11-12 14:00)"

Gunakan rentang waktu untuk menjelaskan tren, misalnya:
"Pada periode 2024-11-12 14:00–15:00 (SNAP_ID 3450–3451), CPU usage meningkat tajam."
"""

    final_prompt = f"""
{lang_intro}

{analysis_block}

{rec_block}

{style_block}

{structure_block}

{timestamp_rules}

"""

    return final_prompt