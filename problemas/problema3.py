import numpy as np
import os
import pandas as pd

os.makedirs('../imagenes', exist_ok=True)

a, c, m, seed = 16807, 0, 2147483647, 22779

def lcg_bits(a, c, m, seed, total_bits):
    xi = int(seed)
    out = np.empty(total_bits, dtype=np.int8)
    for i in range(total_bits):
        xi = (a * xi + c) % m
        out[i] = xi & 1
    return out

total_bits = 1_000_000

bits_lcg_array = lcg_bits(a, c, m, seed, total_bits)
rng = np.random.default_rng(123456)
bits_mt_array = rng.integers(0, 2, size=total_bits, dtype=np.int8)

bits_lcg_array = np.ascontiguousarray(bits_lcg_array, dtype=np.int8)
bits_mt_array = np.ascontiguousarray(bits_mt_array, dtype=np.int8)

print(f"Bits generados: LCG={bits_lcg_array.size}, MT={bits_mt_array.size}")

results = {}

try:
    from nistrng.sp800_22r1a import (
        MonobitTest,
        FrequencyWithinBlockTest,
        RunsTest,
        LongestRunOnesInABlockTest,
        BinaryMatrixRankTest,
        DiscreteFourierTransformTest,
        NonOverlappingTemplateMatchingTest,
        OverlappingTemplateMatchingTest,
        MaurersUniversalTest,
        LinearComplexityTest,
        SerialTest,
        ApproximateEntropyTest,
        CumulativeSumsTest,
        RandomExcursionTest,
        RandomExcursionVariantTest
    )

    battery = {
        "monobit": MonobitTest(),
        "frequency_within_block": FrequencyWithinBlockTest(),
        "runs": RunsTest(),
        "longest_run_ones_in_a_block": LongestRunOnesInABlockTest(),
        "binary_matrix_rank": BinaryMatrixRankTest(),
        "dft": DiscreteFourierTransformTest(),
        "non_overlapping_template_matching": NonOverlappingTemplateMatchingTest(),
        "overlapping_template_matching": OverlappingTemplateMatchingTest(),
        "maurers_universal": MaurersUniversalTest(),
        "linear_complexity": LinearComplexityTest(),
        "serial": SerialTest(),
        "approximate_entropy": ApproximateEntropyTest(),
        "cumulative_sums": CumulativeSumsTest(),
        "random_excursion": RandomExcursionTest(),
        "random_excursion_variant": RandomExcursionVariantTest()
    }

    test_name_mapping = {
        "monobit": "frequency",
        "frequency_within_block": "block_frequency",
        "runs": "runs",
        "longest_run_ones_in_a_block": "longest_run_of_ones",
        "binary_matrix_rank": "rank",
        "dft": "spectral",
        "non_overlapping_template_matching": "non_overlapping_template",
        "overlapping_template_matching": "overlapping_template",
        "maurers_universal": "universal",
        "linear_complexity": "linear_complexity",
        "serial": "serial",
        "approximate_entropy": "approximate_entropy",
        "cumulative_sums": "cumulative_sums",
        "random_excursion": "random_excursions",
        "random_excursion_variant": "random_excursions_variant"
    }

    def extract_pvalue(res):
        """
        Extrae el p-valor de diferentes formatos de resultado de nistrng.
        El formato típico es: (Result_object, score) 
        donde Result_object tiene _score_list que contiene el p-valor
        """
        if res is None:
            return None
        
        # Si es directamente un número válido
        if isinstance(res, (float, int, np.floating, np.integer)):
            val = float(res)
            if 0 <= val <= 1:
                return val
            return None
        
        # Si es una tupla/lista - nistrng devuelve (Result_object, score)
        if isinstance(res, (list, tuple)) and len(res) > 0:
            # Intentar extraer del primer elemento (el objeto Result)
            first_pval = extract_pvalue(res[0])
            if first_pval is not None:
                return first_pval
            # Si no funciona, intentar con los demás elementos
            for item in res[1:]:
                pv = extract_pvalue(item)
                if pv is not None:
                    return pv
            return None
        
        # Si tiene atributo _score_list (caso de nistrng.test.Result)
        if hasattr(res, "_score_list"):
            score_list = getattr(res, "_score_list")
            return extract_pvalue(score_list)
        
        # Si tiene atributo p_value
        if hasattr(res, "p_value"):
            return extract_pvalue(getattr(res, "p_value"))
        
        # Si tiene atributo pvalue
        if hasattr(res, "pvalue"):
            return extract_pvalue(getattr(res, "pvalue"))
        
        # Si es un diccionario
        if isinstance(res, dict):
            # Buscar claves comunes
            for key in ['_score_list', 'p_value', 'pvalue', 'p', 'pval']:
                if key in res:
                    pv = extract_pvalue(res[key])
                    if pv is not None:
                        return pv
            # Buscar en todos los valores
            for v in res.values():
                pv = extract_pvalue(v)
                if pv is not None:
                    return pv
            return None
        
        # Si es un array numpy
        try:
            arr = np.asarray(res)
            if arr.size == 0:
                return None
            val = float(arr.flat[0])
            if 0 <= val <= 1:
                return val
        except Exception:
            pass
        
        return None

    print("\nEjecutando tests NIST SP 800-22:")
    print("="*90)

    for test_key, test_obj in battery.items():
        test_name = test_name_mapping.get(test_key, test_key)
        try:
            # Ejecutar test en LCG
            result_lcg = test_obj.run(bits_lcg_array)
            p_lcg = extract_pvalue(result_lcg)

            # Ejecutar test en MT
            result_mt = test_obj.run(bits_mt_array)
            p_mt = extract_pvalue(result_mt)

            results[test_name] = {'lcg': p_lcg, 'mt': p_mt}

            # Mostrar resultados
            lcg_status = "PASS" if (p_lcg is not None and p_lcg >= 0.01) else "FAIL"
            mt_status = "PASS" if (p_mt is not None and p_mt >= 0.01) else "FAIL"

            p_lcg_str = f"{p_lcg:8.6f}" if p_lcg is not None else "   None"
            p_mt_str = f"{p_mt:8.6f}" if p_mt is not None else "   None"

            print(f"  {test_name:30s}: LCG={p_lcg_str} ({lcg_status})  MT={p_mt_str} ({mt_status})")

        except Exception as e:
            print(f"\n  {test_name:30s}: Error - {str(e)[:200]}")
            results[test_name] = {'lcg': None, 'mt': None}
            import traceback
            traceback.print_exc()

    print("\n" + "="*90)

except ImportError as ie:
    print(f"Error: nistrng no instalado correctamente.")
    print(f"Ejecuta: pip install nistrng")
    print(f"Detalle: {ie}")
except Exception as e:
    print(f"Error al ejecutar tests: {str(e)}")
    import traceback
    traceback.print_exc()

# Generar reporte
if results:
    rows = []
    for test_name in sorted(results.keys()):
        rows.append({
            'test': test_name,
            'pvalue_lcg': results[test_name]['lcg'],
            'pvalue_mt': results[test_name]['mt']
        })
    df = pd.DataFrame(rows)

    df.to_csv('../imagenes/problema3_nist_results.csv', index=False)
    
    print("\n" + "="*90)
    print("RESULTADOS GUARDADOS")
    print("="*90)
    print(df.to_string(index=False))
    print(f"\n✓ Archivo: ../imagenes/problema3_nist_results.csv")

    # Análisis
    alpha = 0.01
    lcg_passed = 0
    mt_passed = 0
    total_tests = 0

    print("\n" + "="*90)
    print(f"ANÁLISIS COMPARATIVO (α = {alpha})")
    print("="*90)
    print(f"{'Test':<30} {'LCG':<10} {'MT':<10}")
    print("-"*90)

    for _, row in df.iterrows():
        if pd.notna(row['pvalue_lcg']) and pd.notna(row['pvalue_mt']):
            total_tests += 1
            lcg_result = "PASS" if row['pvalue_lcg'] >= alpha else "FAIL"
            mt_result = "PASS" if row['pvalue_mt'] >= alpha else "FAIL"

            if row['pvalue_lcg'] >= alpha:
                lcg_passed += 1
            if row['pvalue_mt'] >= alpha:
                mt_passed += 1

            print(f"{row['test']:<30} {lcg_result:<10} {mt_result:<10}")

    if total_tests > 0:
        print("\n" + "="*90)
        print("RESUMEN")
        print("="*90)
        lcg_pct = (lcg_passed/total_tests*100)
        mt_pct = (mt_passed/total_tests*100)
        
        print(f"LCG:              {lcg_passed}/{total_tests} tests pasados ({lcg_pct:.1f}%)")
        print(f"Mersenne Twister: {mt_passed}/{total_tests} tests pasados ({mt_pct:.1f}%)")

        print("\n" + "="*90)
        print("CONCLUSIÓN")
        print("="*90)
        
        if mt_passed > lcg_passed:
            diff = mt_passed - lcg_passed
            print(f"✓ MERSENNE TWISTER tiene MEJOR desempeño")
            print(f"  - Pasa {diff} tests más que LCG ({mt_passed} vs {lcg_passed})")
            print(f"  - Diferencia de {mt_pct - lcg_pct:.1f} puntos porcentuales")
        elif lcg_passed > mt_passed:
            diff = lcg_passed - mt_passed
            print(f"✓ LCG tiene MEJOR desempeño")
            print(f"  - Pasa {diff} tests más que MT ({lcg_passed} vs {mt_passed})")
            print(f"  - Diferencia de {lcg_pct - mt_pct:.1f} puntos porcentuales")
        else:
            print(f"✓ Ambos generadores tienen desempeño SIMILAR")
            print(f"  - Ambos pasan {lcg_passed} tests")
        
        print("\nInterpretación:")
        if mt_pct >= 90:
            print("  MT: ★★★★★ Excelente calidad (cryptographically secure)")
        elif mt_pct >= 80:
            print("  MT: ★★★★☆ Muy buena calidad")
        elif mt_pct >= 70:
            print("  MT: ★★★☆☆ Buena calidad")
        elif mt_pct >= 50:
            print("  MT: ★★☆☆☆ Calidad aceptable")
        else:
            print("  MT: ★☆☆☆☆ Calidad cuestionable")
            
        if lcg_pct >= 90:
            print("  LCG: ★★★★★ Excelente calidad (inesperado para LCG)")
        elif lcg_pct >= 80:
            print("  LCG: ★★★★☆ Muy buena calidad")
        elif lcg_pct >= 70:
            print("  LCG: ★★★☆☆ Buena calidad")
        elif lcg_pct >= 50:
            print("  LCG: ★★☆☆☆ Calidad aceptable")
        else:
            print("  LCG: ★☆☆☆☆ Calidad baja (típico para LCG simple)")
        
        print("="*90)
    else:
        print("\n⚠ No se pudieron completar tests válidos")
        print("   Los p-valores extraídos no están en el rango [0, 1]")
else:
    print("\n❌ No se pudieron ejecutar los tests NIST.")
    print("   Instala nistrng: pip install nistrng")