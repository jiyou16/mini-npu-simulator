"""Mini NPU 시뮬레이터 — MAC(Multiply-Accumulate) 연산 기반 패턴 판별기.

실행:
    python main.py                       # 대화형 메뉴
    python main.py --mode 1              # 모드 1: 3×3 직접 입력
    python main.py --mode 2 [--data data.json] [--repeat 10]
                                         # 모드 2: JSON 일괄 판정 + 성능 분석 + 리포트

외부 라이브러리(NumPy 등)는 사용하지 않는다. 표준 라이브러리만 사용.
"""

import argparse
import json
import re
import sys
import time

EPSILON = 1e-9                      # 점수 동점 판정 허용오차
LABEL_CROSS = "Cross"               # 내부 표준 라벨
LABEL_X = "X"
LABEL_UNDECIDED = "UNDECIDED"
PATTERN_KEY_RE = re.compile(r"^size_(\d+)_(\d+)$")
FILTER_KEY_RE = re.compile(r"^size_(\d+)$")
PERF_SIZES = (3, 5, 13, 25)
DEFAULT_REPEAT = 10


# ---------------------------------------------------------------------------
# 데이터 구조
# ---------------------------------------------------------------------------
class Grid:
    """n×n 2차원 숫자 배열. 특정 위치의 값을 저장(set)하고 읽을(get) 수 있다."""

    def __init__(self, n, fill=0.0):
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"크기는 양의 정수여야 합니다: {n!r}")
        self.n = n
        self._cells = [[float(fill)] * n for _ in range(n)]

    def get(self, r, c):
        return self._cells[r][c]

    def set(self, r, c, value):
        self._cells[r][c] = float(value)

    def rows(self):
        return [row[:] for row in self._cells]

    @classmethod
    def from_rows(cls, rows):
        """2차원 리스트 → Grid. 정사각형이 아니거나 숫자가 아니면 ValueError."""
        if not isinstance(rows, list) or not rows:
            raise ValueError("2차원 배열(비어 있지 않은 리스트)이 아닙니다")
        n = len(rows)
        grid = cls(n)
        for r, row in enumerate(rows):
            if not isinstance(row, list):
                raise ValueError(f"{r + 1}행이 리스트가 아닙니다")
            if len(row) != n:
                raise ValueError(f"{r + 1}행의 열 수 {len(row)} ≠ 행 수 {n} (정사각형 아님)")
            for c, v in enumerate(row):
                # bool은 int의 하위 타입이므로 명시적으로 거부
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(f"({r + 1},{c + 1}) 값이 숫자가 아닙니다: {v!r}")
                grid.set(r, c, v)
        return grid

    def __repr__(self):
        return f"Grid({self.n}×{self.n})"


# ---------------------------------------------------------------------------
# MAC 연산 / 비교 정책 / 라벨 정규화
# ---------------------------------------------------------------------------
def mac(pattern, filt):
    """위치별 곱셈 후 누적 합. 반복문으로 직접 구현 (곱셈 N²회)."""
    if pattern.n != filt.n:
        raise ValueError(f"크기 불일치: 패턴 {pattern.n}×{pattern.n}, 필터 {filt.n}×{filt.n}")
    total = 0.0
    n = pattern.n
    for r in range(n):
        for c in range(n):
            total += pattern.get(r, c) * filt.get(r, c)
    return total


def compare_scores(score_a, score_b, eps=EPSILON):
    """허용오차 기반 비교. 'A' / 'B' / None(동점)."""
    if abs(score_a - score_b) < eps:
        return None
    return "A" if score_a > score_b else "B"


_LABEL_MAP = {
    "+": LABEL_CROSS, "cross": LABEL_CROSS,
    "x": LABEL_X,
}


def normalize_label(raw):
    """expected 값('+', 'x')과 필터 키('cross', 'x')를 표준 라벨(Cross/X)로. 실패 시 None."""
    if not isinstance(raw, str):
        return None
    return _LABEL_MAP.get(raw.strip().lower())


# ---------------------------------------------------------------------------
# 모드 1: 3×3 콘솔 입력
# ---------------------------------------------------------------------------
def read_grid_from_console(title, n=3):
    print(f"\n[{title}] {n}×{n} 값을 한 줄씩 공백으로 구분해 입력하세요.")
    while True:
        rows = []
        ok = True
        for r in range(n):
            line = input(f"  {r + 1}행> ")
            parts = line.split()
            if len(parts) != n:
                print(f"입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요. "
                      f"(입력된 값 {len(parts)}개)")
                ok = False
                break
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                print(f"입력 형식 오류: 숫자로 변환할 수 없는 값이 있습니다 → {line.strip()!r}")
                ok = False
                break
        if ok:
            return Grid.from_rows(rows)
        print(f"  ↳ [{title}]를 처음부터 다시 입력하세요.")


def run_mode1():
    print("\n=== 모드 1: 3×3 사용자 입력 ===")
    filter_a = read_grid_from_console("필터 A")
    filter_b = read_grid_from_console("필터 B")
    pattern = read_grid_from_console("패턴")

    start = time.perf_counter()
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    elapsed_ms = (time.perf_counter() - start) * 1000

    winner = compare_scores(score_a, score_b)
    verdict = {"A": "A", "B": "B", None: "판정 불가"}[winner]

    print("\n--- 결과 ---")
    print(f"필터 A 점수 : {score_a:g}")
    print(f"필터 B 점수 : {score_b:g}")
    print(f"연산 시간   : {elapsed_ms:.4f} ms (MAC 2회, 곱셈 {2 * 9}회)")
    print(f"판정 결과   : {verdict}"
          + (f"  (|A−B| < {EPSILON:g} → 동점)" if winner is None else ""))


# ---------------------------------------------------------------------------
# 모드 2: data.json 로드·검증·일괄 판정
# ---------------------------------------------------------------------------
def load_filters(raw_filters):
    """filters 섹션 → {N: {'Cross': Grid, 'X': Grid}}, 경고 목록."""
    filters, warnings = {}, []
    if not isinstance(raw_filters, dict):
        return filters, ["'filters'가 객체(dict)가 아닙니다"]
    for size_key, group in raw_filters.items():
        m = FILTER_KEY_RE.match(str(size_key).strip())
        if not m:
            warnings.append(f"필터 키 '{size_key}' 형식 오류 (size_N 이어야 함) → 무시")
            continue
        n = int(m.group(1))
        if not isinstance(group, dict):
            warnings.append(f"'{size_key}' 값이 객체가 아닙니다 → 무시")
            continue
        entry = {}
        for label_key, rows in group.items():
            label = normalize_label(label_key)
            if label is None:
                warnings.append(f"'{size_key}'의 필터 라벨 '{label_key}' 인식 불가 → 무시")
                continue
            try:
                grid = Grid.from_rows(rows)
            except ValueError as e:
                warnings.append(f"'{size_key}.{label_key}' 필터 오류: {e} → 무시")
                continue
            if grid.n != n:
                warnings.append(f"'{size_key}.{label_key}' 실제 크기 {grid.n}×{grid.n} ≠ 키의 N={n} → 무시")
                continue
            entry[label] = grid
        filters[n] = entry
    return filters, warnings


def evaluate_case(key, case, filters):
    """패턴 1건 판정. 예외를 밖으로 던지지 않고 결과 dict로 반환."""
    res = {"key": key, "score_cross": None, "score_x": None,
           "verdict": None, "expected": None, "status": "FAIL", "reason": ""}
    try:
        m = PATTERN_KEY_RE.match(str(key).strip())
        if not m:
            raise ValueError("패턴 키 형식 오류 (size_{N}_{idx} 이어야 함)")
        n = int(m.group(1))
        if not isinstance(case, dict):
            raise ValueError("패턴 항목이 객체가 아닙니다")
        if "input" not in case:
            raise ValueError("'input' 필드 누락")
        if "expected" not in case:
            raise ValueError("'expected' 필드 누락")

        expected = normalize_label(case["expected"])
        res["expected"] = expected
        if expected is None:
            raise ValueError(f"expected 라벨 인식 불가: {case['expected']!r} ('+' 또는 'x' 필요)")

        if n not in filters:
            raise ValueError(f"size_{n} 필터가 없습니다")
        group = filters[n]
        missing = [lbl for lbl in (LABEL_CROSS, LABEL_X) if lbl not in group]
        if missing:
            raise ValueError(f"size_{n} 필터에 {', '.join(missing)} 필터가 없습니다")

        pattern = Grid.from_rows(case["input"])
        if pattern.n != n:
            raise ValueError(f"크기 불일치: 키는 N={n}, 입력 패턴은 {pattern.n}×{pattern.n}")

        sc = mac(pattern, group[LABEL_CROSS])
        sx = mac(pattern, group[LABEL_X])
        res["score_cross"], res["score_x"] = sc, sx
        winner = compare_scores(sc, sx)
        res["verdict"] = {"A": LABEL_CROSS, "B": LABEL_X, None: LABEL_UNDECIDED}[winner]

        if res["verdict"] == expected:
            res["status"] = "PASS"
        elif res["verdict"] == LABEL_UNDECIDED:
            res["reason"] = f"동점(|Δ| < {EPSILON:g}) → UNDECIDED, expected={expected}"
        else:
            res["reason"] = f"판정 {res['verdict']} ≠ expected {expected}"
    except (ValueError, TypeError) as e:
        res["reason"] = str(e)
    return res


def fmt_score(v):
    return "-" if v is None else f"{v:.6g}"


def run_batch(data, filters):
    patterns = data.get("patterns")
    if not isinstance(patterns, dict):
        print("오류: 'patterns'가 객체(dict)가 아닙니다.")
        return []

    print(f"\n{'케이스':<14}{'Cross 점수':>12}{'X 점수':>12}  {'판정':<10}{'expected':<10}결과")
    print("-" * 72)
    results = []
    for key, case in patterns.items():
        r = evaluate_case(key, case, filters)
        results.append(r)
        print(f"{key:<14}{fmt_score(r['score_cross']):>12}{fmt_score(r['score_x']):>12}  "
              f"{(r['verdict'] or '-'):<10}{(r['expected'] or '-'):<10}{r['status']}"
              + (f"  ← {r['reason']}" if r["status"] == "FAIL" else ""))
    return results


# ---------------------------------------------------------------------------
# 성능 분석
# ---------------------------------------------------------------------------
def make_cross(n):
    g = Grid(n)
    mid = n // 2
    for i in range(n):
        g.set(mid, i, 1)
        g.set(i, mid, 1)
    return g


def make_x(n):
    g = Grid(n)
    for i in range(n):
        g.set(i, i, 1)
        g.set(i, n - 1 - i, 1)
    return g


def measure_mac(pattern, filt, repeat):
    mac(pattern, filt)                      # 워밍업 1회 (측정 제외)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        mac(pattern, filt)
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times), min(times)


def run_performance(filters=None, repeat=DEFAULT_REPEAT):
    filters = filters or {}
    print(f"\n=== 성능 분석 (MAC 1회 기준, 워밍업 1회 + 반복 {repeat}회) ===")
    print(f"{'크기':<8}{'연산 횟수 N²':>12}{'평균(ms)':>12}{'최소(ms)':>12}{'평균/N²(µs)':>14}  필터 출처")
    print("-" * 72)
    rows = []
    for n in PERF_SIZES:
        filt = filters.get(n, {}).get(LABEL_CROSS)
        source = "data.json" if filt is not None else "자동 생성"
        if filt is None:
            filt = make_cross(n)
        pattern = make_x(n)
        avg, best = measure_mac(pattern, filt, repeat)
        ops = n * n
        rows.append((n, ops, avg))
        print(f"{n}×{n:<5}{ops:>12}{avg:>12.5f}{best:>12.5f}{avg * 1000 / ops:>14.4f}  {source}")
    if len(rows) >= 2:
        (n0, o0, t0), (n1, o1, t1) = rows[0], rows[-1]
        if t0 > 0:
            print(f"\n{n0}×{n0} → {n1}×{n1}: 연산 횟수 {o1 / o0:.1f}배, 평균 시간 {t1 / t0:.1f}배 "
                  f"(작은 크기는 호출 오버헤드 비중이 커서 비율이 더 작게 나올 수 있음)")
    return rows


# ---------------------------------------------------------------------------
# 결과 리포트
# ---------------------------------------------------------------------------
def print_report(results, warnings):
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = [r for r in results if r["status"] == "FAIL"]
    print("\n=== 결과 리포트 ===")
    print(f"전체 테스트: {total}  |  통과: {passed}  |  실패: {len(failed)}")
    if failed:
        print("실패 케이스:")
        for r in failed:
            print(f"  - {r['key']}: {r['reason']}")
    else:
        print("실패 케이스: 없음")
    if warnings:
        print("필터 로드 경고:")
        for w in warnings:
            print(f"  - {w}")


def run_mode2(path="data.json", repeat=DEFAULT_REPEAT):
    print(f"\n=== 모드 2: JSON 데이터 분석 ({path}) ===")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 → {path}")
        return None
    except json.JSONDecodeError as e:
        print(f"오류: JSON 파싱 실패 → {e}")
        return None
    if not isinstance(data, dict):
        print("오류: 최상위 JSON이 객체가 아닙니다.")
        return None

    filters, warnings = load_filters(data.get("filters"))
    loaded = ", ".join(f"size_{n}({'/'.join(sorted(g))})" for n, g in sorted(filters.items()))
    print(f"로드된 필터: {loaded or '없음'}")
    for w in warnings:
        print(f"  경고: {w}")

    results = run_batch(data, filters)
    run_performance(filters, repeat)
    print_report(results, warnings)
    return results


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Mini NPU 시뮬레이터")
    parser.add_argument("--mode", choices=["1", "2"], help="1: 3×3 입력, 2: JSON 분석")
    parser.add_argument("--data", default="data.json", help="모드 2 JSON 경로")
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT, help="성능 측정 반복 횟수")
    args = parser.parse_args()
    repeat = max(1, args.repeat)

    try:
        if args.mode == "1":
            run_mode1()
            return
        if args.mode == "2":
            run_mode2(args.data, repeat)
            return
        while True:
            print("\n===== Mini NPU 시뮬레이터 =====")
            print("1. 3×3 직접 입력 (필터 A/B + 패턴)")
            print("2. data.json 일괄 판정 + 성능 분석 + 리포트")
            print("0. 종료")
            choice = input("선택> ").strip()
            if choice == "1":
                run_mode1()
            elif choice == "2":
                path = input(f"JSON 경로 (Enter = {args.data})> ").strip() or args.data
                run_mode2(path, repeat)
            elif choice == "0":
                print("종료합니다.")
                return
            else:
                print("0, 1, 2 중에서 선택하세요.")
    except (KeyboardInterrupt, EOFError):
        print("\n입력이 중단되어 종료합니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()
