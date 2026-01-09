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
2. Temuan Utama
3. Akar Masalah
4. Analisis Tren (CPU, Wait, AAS, I/O, RAC)
"""
    elif analysis_level == "technical":
        analysis_block = """
Berikan analisis teknis lengkap dan terstruktur:
- Bahas CPU, wait events, AAS, I/O, RAC, dan SQL.
- Identifikasi bottleneck utama dan pola antar snapshot.

Wajib sertakan:
1. Ringkasan Status
2. Temuan Utama
3. Akar Masalah
4. Analisis Tren Detail (CPU, Wait, AAS, I/O, RAC, SQL)
"""
    else:  # deepdive
        analysis_block = """
Berikan analisis tingkat ahli dengan kedalaman maksimal:
- Bahas parameter Oracle yang relevan.
- Jelaskan perilaku optimizer, concurrency, GC, latch, dan I/O subsystem.

Wajib sertakan:
1. Ringkasan Status
2. Temuan Utama
3. Akar Masalah
4. Analisis Mendalam (CPU, Wait, AAS, I/O, RAC, SQL)
"""

    # ============================
    # 3. Level Rekomendasi
    # ============================
    if recommendation_level == "high":
        rec_block = """
Berikan rekomendasi tingkat tinggi:
1. Rekomendasi Tindakan (3–5 poin)
2. Prioritas
"""
    elif recommendation_level == "medium":
        rec_block = """
Berikan rekomendasi teknis tingkat menengah:
1. Rekomendasi Tindakan (5–8 poin)
2. Prioritas
"""
    else:  # expert
        rec_block = """
Berikan rekomendasi tuning tingkat ahli:
1. Rekomendasi Tindakan (8–12 poin)
2. Prioritas
3. Catatan Teknis
"""

    # ============================
    # 4. Style (Hybrid)
    # ============================
    style_block = """
Gunakan gaya hybrid:
- Struktur laporan mengikuti AWR.
- Insight mendalam mengikuti ADDM.
- Tambahkan interpretasi AI untuk menjelaskan pola dan korelasi.
"""

    # ============================
    # 5. Struktur laporan wajib
    # ============================
    structure_block = """
Struktur laporan wajib:
1. Executive Summary
2. Ringkasan Status
3. Temuan Utama
4. CPU Trend Analysis
5. Wait Event Analysis
6. AAS Trend Analysis
7. I/O Analysis
8. RAC Analysis
9. Top SQL Analysis
10. Akar Masalah
11. Rekomendasi Tindakan
"""

    # ============================
    # 6. Timestamp Rules
    # ============================
    timestamp_rules = """
Saat menyebutkan SNAP_ID, selalu sertakan timestamp:
Format: YYYY-MM-DD HH24:MI
Contoh: SNAP_ID 3450 (2024-11-12 14:00)
"""

    # ============================
    # FINAL PROMPT
    # ============================
    final_prompt = f"""
{lang_intro}

{analysis_block}

{rec_block}

{style_block}

{structure_block}

{timestamp_rules}
"""

    return final_prompt