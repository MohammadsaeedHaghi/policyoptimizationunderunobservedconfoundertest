# Sources and provenance

| dataset | file(s) | source | verified |
|---|---|---|---|
| IHDP | `ihdp/ihdp_npci_1-100.{train,test}.npz` | https://www.fredjo.com/files/ | 672x25x100 train, 75 test |
| Twins | `twins/twin_pairs_{X,T,Y}_3years_samesex.csv` | CEVAE repo, `AMLab-Amsterdam/CEVAE` | 71,345 pairs x 54 |
| LaLonde/NSW | `lalonde/nsw_dw.dta`, `nswre74_*.txt`, `cps_controls.txt`, `psid_controls.txt` | https://users.nber.org/~rdehejia/data/ | 185 treated + 260 control |
| NHEFS | `nhefs/nhefs.csv` | `causaldata` PyPI package | 1,629 x 67 |
| IST | `ist/IST_corrected.csv` | https://datashare.ed.ac.uk/ (10283/124) | 19,435 x 112, latin-1 |
| STAR | `star/STAR.csv` | Rdatasets mirror of R `AER::STAR` | 11,598 x 48 |
| OHIE | **NOT DOWNLOADED** | data.nber.org returns the MyNBER login page | needs free NBER registration |

Two sources in the original plan did not work and were replaced:
- The Harvard CDN NHEFS link now serves a WordPress page, not the CSV -> used the `causaldata` package.
- The GANITE mirror of Twins is 404 -> used the CEVAE repository copy.

`IST_corrected.csv` contains non-UTF8 bytes; read it with `encoding="latin-1"`.
