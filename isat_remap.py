"""
ISAT Category Remapping Tool

掃描 ISAT 資料夾的種類標注，支援改名、合併、刪除，安全輸出至新資料夾。

Usage:
    # 互動式（查看後逐步操作）
    poetry run python isat_remap.py --source folder_A

    # 直接帶參數
    poetry run python isat_remap.py --source folder_A --rename blade=knife
    poetry run python isat_remap.py --source folder_A --merge knife,blade=weapon
    poetry run python isat_remap.py --source folder_A --delete __background__

    # 組合操作
    poetry run python isat_remap.py --source folder_A \\
        --rename blade=knife \\
        --delete __background__

    # 預覽不寫入
    poetry run python isat_remap.py --source folder_A --rename blade=knife --dry-run

    # 覆寫原始資料（謹慎）
    poetry run python isat_remap.py --source folder_A --rename blade=knife --inplace
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def scan_categories(source: Path) -> dict[str, dict]:
    """Return {category: {count, files}} across all ISAT JSONs in source."""
    stats: dict[str, dict] = defaultdict(lambda: {"count": 0, "files": set()})
    for jf in sorted(source.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("info", {}).get("description") != "ISAT":
            continue
        for obj in data.get("objects", []):
            cat = obj.get("category", "").strip()
            if cat:
                stats[cat]["count"] += 1
                stats[cat]["files"].add(jf.name)
    return dict(stats)


def print_categories(stats: dict[str, dict]) -> None:
    if not stats:
        print("  (沒有找到任何種類)")
        return
    width = max(len(c) for c in stats)
    print(f"\n  {'種類':<{width}}  物件數  圖片數")
    print(f"  {'-'*width}  ------  ------")
    for i, (cat, info) in enumerate(sorted(stats.items())):
        print(f"  [{i}] {cat:<{width}}  {info['count']:>6}  {len(info['files']):>6}")
    print()


# ---------------------------------------------------------------------------
# Build remap table
# ---------------------------------------------------------------------------

def build_remap(
    stats: dict[str, dict],
    renames: list[str],
    merges: list[str],
    deletes: list[str],
) -> tuple[dict[str, str | None], list[str]]:
    """Return (remap_table, errors).

    remap_table: {old_cat: new_cat}  — None means delete
    """
    existing = set(stats.keys())
    remap: dict[str, str | None] = {}
    errors: list[str] = []

    for r in renames:
        if "=" not in r:
            errors.append(f"--rename 格式錯誤（需要 old=new）: {r!r}")
            continue
        old, new = r.split("=", 1)
        old, new = old.strip(), new.strip()
        if old not in existing:
            errors.append(f"--rename: 找不到種類 {old!r}（現有: {sorted(existing)}）")
            continue
        remap[old] = new

    for m in merges:
        if "=" not in m:
            errors.append(f"--merge 格式錯誤（需要 A,B=new）: {m!r}")
            continue
        sources_str, new = m.split("=", 1)
        new = new.strip()
        for src in sources_str.split(","):
            src = src.strip()
            if src not in existing:
                errors.append(f"--merge: 找不到種類 {src!r}（現有: {sorted(existing)}）")
                continue
            remap[src] = new

    for d in deletes:
        d = d.strip()
        if d not in existing:
            errors.append(f"--delete: 找不到種類 {d!r}（現有: {sorted(existing)}）")
            continue
        remap[d] = None  # None = delete

    return remap, errors


# ---------------------------------------------------------------------------
# Apply remap
# ---------------------------------------------------------------------------

def apply_remap(
    source: Path,
    remap: dict[str, str | None],
    out_dir: Path,
    dry_run: bool,
) -> None:
    """Copy all files to out_dir, applying remap to ISAT JSONs."""
    json_files = [jf for jf in sorted(source.glob("*"))
                  if jf.suffix.lower() == ".json"]
    other_files = [f for f in sorted(source.glob("*"))
                   if f.suffix.lower() != ".json" and f.is_file()]

    changed_objs = 0
    deleted_objs = 0
    changed_files = 0

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            print(f"  [SKIP] 無法解析: {jf.name}")
            continue

        if data.get("info", {}).get("description") != "ISAT":
            # Non-ISAT json: just copy as-is
            if not dry_run:
                shutil.copy2(jf, out_dir / jf.name)
            continue

        original_objects = data.get("objects", [])
        new_objects = []
        file_changed = False

        for obj in original_objects:
            cat = obj.get("category", "").strip()
            if cat not in remap:
                new_objects.append(obj)
                continue
            new_cat = remap[cat]
            if new_cat is None:
                deleted_objs += 1
                file_changed = True
            else:
                obj = dict(obj)
                obj["category"] = new_cat
                new_objects.append(obj)
                changed_objs += 1
                file_changed = True

        if file_changed:
            changed_files += 1

        data["objects"] = new_objects

        if not dry_run:
            out_path = out_dir / jf.name
            out_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # Copy non-JSON files (images, etc.)
    if not dry_run:
        for f in other_files:
            shutil.copy2(f, out_dir / f.name)

    print(f"  影響檔案: {changed_files} 個 JSON")
    print(f"  改名物件: {changed_objs} 個")
    print(f"  刪除物件: {deleted_objs} 個")
    if dry_run:
        print("  [dry-run] 未寫入任何檔案")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive(source: Path, stats: dict[str, dict]) -> dict[str, str | None]:
    """Walk the user through building a remap table interactively."""
    remap: dict[str, str | None] = {}

    while True:
        print_categories({k: v for k, v in stats.items() if k not in remap})
        if remap:
            print("  目前排定的操作：")
            for old, new in remap.items():
                arrow = "→ [刪除]" if new is None else f"→ {new!r}"
                print(f"    {old!r} {arrow}")
            print()

        op = input("操作 (rename / merge / delete / done / reset): ").strip().lower()

        if op == "done":
            break
        elif op == "reset":
            remap.clear()
        elif op == "rename":
            old = input("  舊種類名稱: ").strip()
            new = input("  新種類名稱: ").strip()
            if old not in stats:
                print(f"  [錯誤] 找不到 {old!r}")
            elif not new:
                print("  [錯誤] 新名稱不可為空")
            else:
                remap[old] = new
        elif op == "merge":
            srcs = [s.strip() for s in input("  要合併的種類（逗號分隔）: ").split(",")]
            new = input("  合併後的種類名稱: ").strip()
            missing = [s for s in srcs if s not in stats]
            if missing:
                print(f"  [錯誤] 找不到種類: {missing}")
            elif not new:
                print("  [錯誤] 名稱不可為空")
            else:
                for s in srcs:
                    remap[s] = new
        elif op == "delete":
            cat = input("  要刪除的種類名稱: ").strip()
            if cat not in stats:
                print(f"  [錯誤] 找不到 {cat!r}")
            else:
                remap[cat] = None
        else:
            print("  [錯誤] 未知操作，請輸入 rename / merge / delete / done / reset")

    return remap


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ISAT 種類屬性管理工具：改名、合併、刪除",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="ISAT 資料夾路徑")
    parser.add_argument(
        "--rename", action="append", default=[], metavar="OLD=NEW",
        help="改名（可多次使用）：--rename knife=weapon",
    )
    parser.add_argument(
        "--merge", action="append", default=[], metavar="A,B=NEW",
        help="合併（可多次使用）：--merge knife,blade=weapon",
    )
    parser.add_argument(
        "--delete", action="append", default=[], metavar="CAT",
        help="刪除種類（可多次使用）：--delete __background__",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只預覽，不寫入任何檔案",
    )
    parser.add_argument(
        "--inplace", action="store_true",
        help="覆寫原始 JSON（預設輸出至 <source>_remapped/）",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.exists():
        print(f"[ERROR] 路徑不存在: {source}")
        sys.exit(1)

    # --- Scan ---
    print(f"\n[INFO] 掃描: {source}")
    stats = scan_categories(source)
    if not stats:
        print("[ERROR] 找不到任何 ISAT JSON 檔案")
        sys.exit(1)

    print(f"[INFO] 找到 {sum(v['count'] for v in stats.values())} 個物件，"
          f"{len(stats)} 種類別")
    print_categories(stats)

    # --- Build remap ---
    interactive_mode = not (args.rename or args.merge or args.delete)

    if interactive_mode:
        remap = interactive(source, stats)
    else:
        remap, errors = build_remap(stats, args.rename, args.merge, args.delete)
        if errors:
            for e in errors:
                print(f"[ERROR] {e}")
            sys.exit(1)

    if not remap:
        print("[INFO] 沒有任何操作，結束。")
        return

    # --- Confirm & output ---
    print("\n將執行以下操作：")
    for old, new in sorted(remap.items()):
        arrow = "→ [刪除]" if new is None else f"→ {new!r}"
        print(f"  {old!r} {arrow}")

    if args.inplace:
        out_dir = source
        print(f"\n[WARNING] --inplace 模式：直接覆寫 {source}")
    else:
        out_dir = source.parent / (source.name + "_remapped")
        print(f"\n輸出目錄: {out_dir}")

    if not args.dry_run and not args.inplace:
        if out_dir.exists():
            print(f"[INFO] 輸出目錄已存在，將覆蓋其中的檔案")
        out_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not args.inplace:
        confirm = input("\n確認執行？(y/N): ").strip().lower()
        if confirm != "y":
            print("取消。")
            return

    print()
    apply_remap(source, remap, out_dir, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\n[完成] 輸出至: {out_dir}")


if __name__ == "__main__":
    main()
