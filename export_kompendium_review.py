
import json, random, csv, argparse, pathlib
import pandas as pd

def load_kompendium(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Expect keys: id/title/short/long; fallback heuristics
            rid = rec.get("id") or rec.get("doc_id") or rec.get("title")
            title = rec.get("title","")
            short = rec.get("short","")
            longt = rec.get("long","") or rec.get("text","")
            items.append({"id": rid, "title": title, "short_text": short, "long_text": longt})
    return items

def main():
    ap = argparse.ArgumentParser(description="Sample Kompendium entries for review.")
    ap.add_argument("--kompendium", required=True, help="Path to kompendium.jsonl")
    ap.add_argument("--k", type=int, default=30, help="Sample size (default 30)")
    ap.add_argument("--out", default="kompendium_review_sample.xlsx", help="Output XLSX file")
    args = ap.parse_args()

    items = load_kompendium(args.kompendium)
    if not items:
        raise SystemExit("No items found in kompendium file.")

    sample = random.sample(items, k=min(args.k, len(items)))

    cols = [
        "id","title","short_text","long_text",
        "accuracy(1-5)","coverage(1-5)","clarity(1-5)","citations(1-5)",
        "pedagogy(1-5)","vocab_alignment(1-5)","style(1-5)","accessibility(1-5)",
        "kg_hooks(1-5, optional)",
        "overall_avg","pass_fail","reviewer","notes"
    ]

    # Build dataframe, prefill first columns from kompendium
    rows = []
    for s in sample:
        row = {c:"" for c in cols}
        row["id"] = s.get("id")
        row["title"] = s.get("title","")
        row["short_text"] = (s.get("short_text") or s.get("short") or "")[:400]
        row["long_text"] = (s.get("long_text") or s.get("long") or s.get("text") or "")[:800]
        rows.append(row)

    df = pd.DataFrame(rows, columns=cols)

    # Write to XLSX and add formulas/validation
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Review")
        ws = writer.book["Review"]

        # overall_avg formula N
        for r in range(2, len(df)+2):
            ws[f"N{r}"] = f"=AVERAGE(E{r}:L{r})"

        from openpyxl.worksheet.datavalidation import DataValidation
        dv = DataValidation(type="list", formula1='"PASS,FAIL"', allow_blank=True, showDropDown=True)
        ws.add_data_validation(dv)
        dv.add(f"O2:O{len(df)+1}")

        # Set column widths
        widths = {
            "A":16,"B":36,"C":48,"D":72,
            "E":14,"F":14,"G":14,"H":14,"I":16,"J":20,"K":14,"L":18,"M":18,
            "N":12,"O":12,"P":18,"Q":48
        }
        for col, w in widths.items():
            ws.column_dimensions[col].width = w

        writer.book.save(args.out)

    print(f"Exported review sample to: {args.out}")

if __name__ == "__main__":
    main()

# How to use the sampling script
# Put your kompendium.jsonl in place (each line one entry with keys like id, title, short, long).
# Run: python export_kompendium_review.py --kompendium /path/to/kompendium.jsonl --k 30 --out kompendium_review_sample.xlsx
# 