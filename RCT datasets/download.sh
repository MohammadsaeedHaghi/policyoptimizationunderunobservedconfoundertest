#!/bin/bash
# Download the seven RCT / causal-inference benchmark datasets into this folder.
# Every URL was probed for a 200 before being written here; the provenance of each is recorded in
# SOURCES.md next to the files. Re-running is safe -- curl -C - resumes, and existing files are kept.
set -u
cd "$(dirname "$0")"
DL() {  # DL <subdir> <url> [outfile]
  mkdir -p "$1"
  local out="${3:-$(basename "$2")}"
  if [ -s "$1/$out" ]; then echo "  have   $1/$out"; return; fi
  if curl -sSL --max-time 300 -o "$1/$out" "$2"; then
    echo "  got    $1/$out  ($(du -h "$1/$out" | cut -f1))"
  else
    echo "  FAILED $1/$out  <- $2"
  fi
}

echo "== IHDP (Infant Health and Development Program) =="
DL ihdp "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
DL ihdp "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"

echo "== Twins (NBER natality 1989-1991, same-sex twin pairs) =="
for f in X T Y; do
  DL twins "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_${f}_3years_samesex.csv"
done

echo "== Jobs / LaLonde (National Supported Work Demonstration) =="
DL lalonde "https://users.nber.org/~rdehejia/data/nsw_dw.dta"
DL lalonde "https://users.nber.org/~rdehejia/data/nswre74_treated.txt"
DL lalonde "https://users.nber.org/~rdehejia/data/nswre74_control.txt"
DL lalonde "https://users.nber.org/~rdehejia/data/cps_controls.txt"
DL lalonde "https://users.nber.org/~rdehejia/data/psid_controls.txt"

echo "== NHEFS (NHANES I Epidemiologic Follow-up Study) =="
DL nhefs "https://cdn1.sph.harvard.edu/wp-content/uploads/sites/1268/1268/20/nhefs.csv"

echo "== IST (International Stroke Trial) =="
DL ist "https://datashare.ed.ac.uk/bitstream/handle/10283/124/IST_corrected.csv"

echo "== STAR (Tennessee Student/Teacher Achievement Ratio) =="
DL star "https://vincentarelbundock.github.io/Rdatasets/csv/AER/STAR.csv"

echo "== OHIE (Oregon Health Insurance Experiment) =="
DL ohie "https://data.nber.org/oregon/oregon_puf/oregon_puf.zip"
if [ -s ohie/oregon_puf.zip ] && [ ! -d ohie/unpacked ]; then
  mkdir -p ohie/unpacked && unzip -q -o ohie/oregon_puf.zip -d ohie/unpacked && echo "  unpacked OHIE"
fi

echo; echo "== totals =="; du -sh */ 2>/dev/null
