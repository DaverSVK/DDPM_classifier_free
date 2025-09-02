#!/usr/bin/env python3
"""
clean_labels.py – keep only label-lines whose images exist
Folders:
    images : DDR/train/
    labels : DDR/train/train.txt
"""

from pathlib import Path

# --- configuration ----------------------------------------------------------
IMG_DIR   = Path("DDR/train")            # folder that contains the .jpg files
LABEL_TXT = IMG_DIR / "train.txt"       # path to labels.txt inside that folder
BACKUP    = True                         # set to False to skip creating a backup
# -----------------------------------------------------------------------------

def main() -> None:
    if not LABEL_TXT.is_file():
        raise FileNotFoundError(f"Label file not found: {LABEL_TXT}")

    original_lines = LABEL_TXT.read_text().splitlines()

    kept, removed = [], []
    for line in original_lines:
        img_name = line.split()[0]          # first token → image file name
        if (IMG_DIR / img_name).is_file():
            kept.append(line)
        else:
            removed.append(line)

    # backup
    if BACKUP:
        LABEL_TXT.with_suffix(".txt.bak").write_text(
            "\n".join(original_lines) + "\n"
        )

    # rewrite labels.txt with the kept lines
    LABEL_TXT.write_text("\n".join(kept) + ("\n" if kept else ""))

    print(f"Done: {len(kept)} kept, {len(removed)} removed.")
    if BACKUP:
        print("Backup saved as", LABEL_TXT.with_suffix(".txt.bak").name)


if __name__ == "__main__":
    main()
