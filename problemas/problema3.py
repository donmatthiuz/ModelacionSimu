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
        if res is None:
            return None
        if isinstance(res, (float, int, np.floating, np.integer)):
            return float(res)
        if isinstance(res, (list, tuple)) and len(res) > 0:
            for item in res:
                pv = extract_pvalue(item)
                if pv is not None:
                    return pv
            return None
        if isinstance(res, dict):
            for v in res.values():
                pv = extract_pvalue(v)
                if pv is not None:
                    return pv
            return None
        if hasattr(res, "p_value"):
            return extract_pvalue(getattr(res, "p_value"))
        if hasattr(res, "pvalues"):
            return extract_pvalue(getattr(res, "pvalues"))
        try:
            arr = np.asarray(res)
            if arr.size == 0:
                return None
            return float(arr.flat[0])
        except Exception:
            return None

    print("Ejecutando tests NIST SP 800-22:")

    for test_key, test_obj in battery.items():
        test_name = test_name_mapping.get(test_key, test_key)
        try:
            # many tests accept numpy arrays of 0/1 as int8/int32
            result_lcg = test_obj.run(bits_lcg_array)
            p_lcg = extract_pvalue(result_lcg)

            result_mt = test_obj.run(bits_mt_array)
            p_mt = extract_pvalue(result_mt)

            results[test_name] = {'lcg': p_lcg, 'mt': p_mt}

            lcg_status = "PASS" if (p_lcg is not None and p_lcg >= 0.01) else "FAIL"
            mt_status = "PASS" if (p_mt is not None and p_mt >= 0.01) else "FAIL"

            p_lcg_str = f"{p_lcg:8.6f}" if p_lcg is not None else "   None"
            p_mt_str = f"{p_mt:8.6f}" if p_mt is not None else "   None"

            print(f"  {test_name:30s}: LCG={p_lcg_str} ({lcg_status})  MT={p_mt_str} ({mt_status})")

        except Exception as e:
            print(f"  {test_name:30s}: Error - {str(e)[:200]}")
            results[test_name] = {'lcg': None, 'mt': None}

    library_used = 'nistrng'

except ImportError:
    print("Error: nistrng no instalado. Ejecuta: pip install nistrng")
    library_used = None
except Exception as e:
    print(f"Error al ejecutar tests: {str(e)}")
    import traceback
    traceback.print_exc()
    library_used = None

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
    #plot()

    print("Resultados guardados en: ../imagenes/problema3_nist_results.csv")
    print(df.to_string(index=False))

    alpha = 0.01
    lcg_passed = 0
    mt_passed = 0
    total_tests = 0

    print(f"Analisis con alpha = {alpha}:")
    print(f"{'Test':<30} {'LCG':<10} {'MT':<10}")

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
        print(f"Resumen:")
        print(f"LCG: {lcg_passed}/{total_tests} tests pasados ({lcg_passed/total_tests*100:.1f}%)")
        print(f"Mersenne Twister: {mt_passed}/{total_tests} tests pasados ({mt_passed/total_tests*100:.1f}%)")

        if mt_passed > lcg_passed:
            print(f"Conclusion: Mersenne Twister tiene mejor desempeño ({mt_passed} vs {lcg_passed} tests pasados)")
        elif lcg_passed > mt_passed:
            print(f"Conclusion: LCG tiene mejor desempeño ({lcg_passed} vs {mt_passed} tests pasados)")
        else:
            print(f"Conclusion: Ambos generadores tienen desempeño similar ({lcg_passed} tests pasados)")
else:
    print("No se pudieron ejecutar los tests NIST.")
    print("Instala nistrng: pip install nistrng")

