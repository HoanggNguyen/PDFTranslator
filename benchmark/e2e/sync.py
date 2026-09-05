"""Đồng bộ artifact giữa máy và một HF **dataset repo** — đường duy nhất.

Bốn pipeline không chạy cùng chỗ: 3 hệ mã nguồn mở chạy trên HF Jobs (để cùng phần
cứng), DeepL chạy ở máy (API thuần, trả tiền GPU để ngồi chờ HTTP là vô nghĩa), còn
người đọc kết quả thì ngồi ở máy. Ba nơi ⇒ phải có **một nguồn sự thật**, và nó là
dataset repo. Máy và job đều chỉ push/pull với nó; không có đường nào máy nói chuyện
trực tiếp với job.

Vì sao dataset repo chứ không phải mount bucket của HF Jobs:

* có ``revision`` ⇒ trỏ được vào đúng phiên bản artifact đã dùng để viết luận văn;
* ``--only`` map thẳng vào ``allow_patterns`` ⇒ kéo riêng ``out/report`` (vài MB)
  thay vì cả ``_render/`` (~250 MB);
* không phụ thuộc ngữ nghĩa mount, nên **cùng một lệnh chạy được ở cả hai phía**;
* khi nộp là có sẵn link tái lập vĩnh viễn.

Bất biến: **chỉ thêm, không xoá.** ``upload_folder`` mặc định không xoá file phía
remote, và ở đây không bao giờ bật ``delete_patterns``. Artifact của một lượt chạy
tốn tiền LLM/GPU/hạn mức DeepL — xoá nhầm là chạy lại từ đầu.

Ví dụ
-----
    # một lần, lúc bắt đầu
    python -m benchmark.e2e.sync init
    python -m benchmark.e2e.sync push --only corpus

    # sau khi DeepL chạy xong ở máy
    python -m benchmark.e2e.sync push --only out/deepl-document

    # trong job HF: kéo bài kiểm về, chạy, đẩy kết quả lên
    python -m benchmark.e2e.sync pull --only corpus
    python -m benchmark.e2e.sync push --only out/babeldoc

    # lấy kết quả về máy
    python -m benchmark.e2e.sync pull --only out/report      # nhẹ
    python -m benchmark.e2e.sync pull                        # đầy đủ
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Thư mục local ↔ tiền tố trên repo. Giữ y hệt tên để đường dẫn trong mọi báo cáo
# đọc được ở cả hai phía mà không phải dịch qua lại.
ROOTS = {
    "corpus": "benchmark/e2e/datasets/corpus",
    "out": "benchmark/e2e/out",
}

# Thứ KHÔNG bao giờ đẩy lên: cache tái tạo được, và file tạm của tiến trình con.
IGNORE = ["*.pyc", "**/__pycache__/**", "**/.DS_Store", "**/_parse/**/*.tmp"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("init", "push", "pull", "ls"))
    p.add_argument("--repo", default=os.environ.get("HF_EVAL_REPO"),
                   help="Dataset repo dạng <user>/<name>. Default: $HF_EVAL_REPO.")
    p.add_argument("--token", default=None,
                   help="Default: $HF_TOKEN. Cần quyền write để push.")
    p.add_argument("--root", type=Path, default=None,
                   help="Gốc repo PDFTranslator. Default: suy từ vị trí file này.")
    p.add_argument("--only", action="append", default=None,
                   help="Giới hạn phạm vi, lặp lại được. Nhận cả tên gốc "
                        "('corpus', 'out') lẫn đường dẫn con ('out/report', "
                        "'out/babeldoc'). Không truyền = toàn bộ.")
    p.add_argument("--revision", default=None,
                   help="Chỉ với pull: kéo đúng một commit/tag. Dùng khi cần dựng "
                        "lại đúng bảng số đã viết trong luận văn.")
    p.add_argument("--private", action="store_true",
                   help="Chỉ với init: tạo repo private (nên bật lúc đang làm).")
    p.add_argument("--message", default=None, help="Commit message khi push.")
    p.add_argument("--dry-run", action="store_true",
                   help="In ra sẽ đẩy/kéo cái gì rồi thoát.")
    return p.parse_args()


def repo_root(explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    # sync.py nằm ở <root>/benchmark/e2e/sync.py
    return Path(__file__).resolve().parents[2]


def resolve_scope(only: list[str] | None) -> list[tuple[str, str]]:
    """``--only`` -> danh sách (đường dẫn local tương đối, tiền tố trên repo).

    Nhận cả ``out`` lẫn ``out/babeldoc``: cái sau cho phép một job chỉ đẩy đúng
    phần nó vừa sinh ra, không đụng artifact của hệ khác đang chạy song song.
    """
    if not only:
        return [(v, k) for k, v in ROOTS.items()]

    scope: list[tuple[str, str]] = []
    for item in only:
        item = item.strip().strip("/")
        head, _, rest = item.partition("/")
        if head not in ROOTS:
            raise SystemExit(f"!! --only '{item}': gốc phải là một trong {sorted(ROOTS)}")
        local = f"{ROOTS[head]}/{rest}" if rest else ROOTS[head]
        remote = f"{head}/{rest}" if rest else head
        scope.append((local, remote))
    return scope


def client(token: str | None):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("!! thiếu huggingface_hub:  pip install huggingface_hub")
    tok = token or os.environ.get("HF_TOKEN") or None
    return HfApi(token=tok), tok


def do_init(api, repo: str, private: bool) -> int:
    from huggingface_hub.errors import HfHubHTTPError

    try:
        api.create_repo(repo_id=repo, repo_type="dataset", private=private,
                        exist_ok=True)
    except HfHubHTTPError as exc:
        print(f"!! không tạo được {repo}: {exc}")
        return 1
    print(f"  dataset repo sẵn sàng: https://huggingface.co/datasets/{repo}"
          f"  ({'private' if private else 'public'})")
    return 0


def do_push(api, repo: str, root: Path, scope: list[tuple[str, str]],
            message: str | None, dry: bool) -> int:
    rc = 0
    for local_rel, remote in scope:
        local = root / local_rel
        if not local.is_dir():
            print(f"  [bỏ qua] {local_rel} chưa tồn tại")
            continue
        n = sum(1 for _ in local.rglob("*") if _.is_file())
        size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
        print(f"  push {local_rel:44} -> {repo}:{remote}  "
              f"({n} file, {size / 1e6:.1f} MB)", flush=True)
        if dry:
            continue
        try:
            api.upload_folder(
                repo_id=repo, repo_type="dataset",
                folder_path=str(local), path_in_repo=remote,
                ignore_patterns=IGNORE,
                commit_message=message or f"push {remote}",
            )
        except Exception as exc:  # noqa: BLE001 — một nhánh hỏng không giết cả lệnh
            print(f"  !! lỗi khi push {remote}: {type(exc).__name__}: {exc}")
            rc = 1
    return rc


def do_pull(repo: str, root: Path, scope: list[tuple[str, str]], token: str | None,
            revision: str | None, dry: bool) -> int:
    from huggingface_hub import snapshot_download

    patterns = [f"{remote}/**" for _, remote in scope]
    print(f"  pull {repo}  patterns={patterns}"
          + (f"  revision={revision}" if revision else ""), flush=True)
    if dry:
        return 0

    # Tải vào một thư mục tạm rồi mới chép sang đúng chỗ: snapshot_download trả về
    # cây theo tiền tố REPO ('corpus/…'), còn ở máy thì corpus nằm sâu trong
    # benchmark/e2e/datasets/. Hai cây khác nhau nên phải map lại từng gốc.
    local_snapshot = snapshot_download(
        repo_id=repo, repo_type="dataset", revision=revision,
        allow_patterns=patterns, token=token,
    )
    snap = Path(local_snapshot)

    import shutil

    n_files = 0
    for local_rel, remote in scope:
        src = snap / remote
        if not src.is_dir():
            print(f"  [bỏ qua] {remote} không có trên repo")
            continue
        dst = root / local_rel
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.rglob("*"):
            if not f.is_file():
                continue
            target = dst / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            n_files += 1
        print(f"  {remote:44} -> {local_rel}")
    print(f"  {n_files} file đã về máy.")
    return 0


def do_ls(api, repo: str) -> int:
    from collections import defaultdict

    files = api.list_repo_files(repo_id=repo, repo_type="dataset")
    if not files:
        print(f"  {repo} rỗng.")
        return 0
    groups: dict[str, int] = defaultdict(int)
    for f in files:
        parts = f.split("/")
        key = "/".join(parts[:3]) if len(parts) >= 3 else "/".join(parts[:2])
        groups[key] += 1
    for key in sorted(groups):
        print(f"  {key:56} {groups[key]:6d} file")
    print(f"  tổng {len(files)} file")
    return 0


def main() -> int:
    args = parse_args()
    if not args.repo:
        print("!! cần --repo hoặc $HF_EVAL_REPO (dạng <user>/pdftranslator-eval)")
        return 1

    root = repo_root(args.root)
    api, token = client(args.token)

    if args.action == "init":
        return do_init(api, args.repo, args.private)
    if args.action == "ls":
        return do_ls(api, args.repo)

    scope = resolve_scope(args.only)
    if args.action == "push":
        if not token:
            print("!! push cần $HF_TOKEN có quyền write")
            return 1
        return do_push(api, args.repo, root, scope, args.message, args.dry_run)
    return do_pull(args.repo, root, scope, token, args.revision, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
