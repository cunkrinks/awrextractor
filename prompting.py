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
            "Gunakan Bahasa Indonesia teknis yang jelas dan profesional. "
            "Istilah Oracle tetap gunakan Bahasa Inggris."
        )
    else:
        lang_intro = (
            "Use clear and professional technical English. "
            "Oracle terminology should remain unchanged."
        )

    # ============================
    # 2. Level Analisis
    # ============================
    if analysis_level == "executive":
        analysis_block = """
Fokus pada ringkasan tingkat tinggi:
- Gambarkan tren besar (CPU, AAS, wait events, I/O).
- Hindari detail teknis mendalam.
- Berikan insight yang mudah dipahami manajemen.
"""
    elif analysis_level == "deepdive":
        analysis_block = """
Berikan analisis teknis mendalam:
- Bahas parameter Oracle yang relevan.
- Jelaskan perilaku optimizer, concurrency, dan I/O subsystem.
- Sertakan insight tingkat ahli seperti GC tuning, latch behavior, dan plan stability.
"""
    else:  # technical
        analysis_block = """
Berikan analisis teknis yang lengkap:
- CPU, wait events, AAS, I/O, RAC.
- Identifikasi bottleneck utama.
- Jelaskan pola dan tren antar snapshot.
"""

    # ============================
    # 3. Level Rekomendasi
    # ============================
    if recommendation_level == "high":
        rec_block = """
Berikan rekomendasi tingkat tinggi:
- Fokus pada area perbaikan umum.
- Hindari parameter spesifik.
"""
    elif recommendation_level == "expert":
        rec_block = """
Berikan rekomendasi tuning tingkat ahli:
- Sertakan parameter Oracle yang relevan.
- Sertakan saran plan stability, parallelism, dan memory tuning.
- Berikan langkah konkret yang dapat dieksekusi DBA senior.
"""
    else:  # medium
        rec_block = """
Berikan rekomendasi teknis tingkat menengah:
- Sertakan insight actionable.
- Hindari parameter terlalu spesifik.
"""

    # ============================
    # 4. Style (Hybrid)
    # ============================
    style_block = """
Gunakan gaya hybrid:
- Struktur laporan mengikuti AWR.
- Insight mendalam mengikuti ADDM.
- Tambahkan interpretasi AI untuk menjelaskan pola dan anomali.
"""

    # ============================
    # 5. Struktur laporan wajib
    # ============================
    structure_block = """
Struktur laporan wajib:
1. Executive Summary
2. CPU Trend Analysis
3. Wait Event Analysis
4. AAS Trend Analysis
5. I/O Analysis
6. RAC Analysis
7. Top SQL Analysis
8. Root Cause Analysis
9. Actionable Recommendations

Jangan membuat angka baru yang tidak ada di konteks.
Jika data tidak tersedia, tulis "data tidak tersedia".
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
"""

    return final_prompt