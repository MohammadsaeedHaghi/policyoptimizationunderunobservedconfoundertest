"""One-off: trim index.html to ONLY the three experiments (First, Second, 2-arm 1-D), removing all other experiment
subtabs+panels. Also makes the Experiment tab the default-open tab and exp_first the default subtab. Backs up first."""
import re, shutil
from pathlib import Path
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
idx=NEW/"index.html"
shutil.copy(idx, NEW/"index.html.bak")
h=idx.read_text()
KEEP={"exp_first","exp_second","exp_1d"}
REMOVE=["exp_c","exp_2arm","exp_1d_uniS","kallus_repro","kallus_methods","exp_nonmono","exp_uncap","exp_3arm","exp_lalonde","exp_ihdp","synthlalonde"]
# 1) remove unwanted subtab buttons (one line each)
alt="|".join(re.escape(r) for r in REMOVE)
h=re.sub(r'(?m)^[ \t]*<button class="subtab-btn[^"]*" onclick="showSub\(\'(?:%s)\',this\)">.*?</button>\n'%alt, "", h)
# 2) remove the three contiguous panel slices (by unique start anchors; keep tab-experiment close)
def cut(h, start_anchor, end_anchor):
    i=h.index(start_anchor); j=h.index(end_anchor); assert i<j, (start_anchor,end_anchor)
    return h[:i]+h[j:]
h=cut(h, '<div id="sub-exp_c" class="subpanel active">', '<div id="sub-exp_1d" class="subpanel">')          # exp_c, exp_2arm
h=cut(h, '<div id="sub-exp_1d_uniS" class="subpanel">', '<div id="sub-exp_first" class="subpanel">')        # uniS..exp_3arm
h=cut(h, '<div id="sub-exp_lalonde" class="subpanel">', '</div>\n\n<script>\nfunction _swap(')               # lalonde,ihdp,synth (keep tab close)
# 3) set defaults: exp_first the active subtab + panel
h=h.replace('<button class="subtab-btn" onclick="showSub(\'exp_first\',this)">',
            '<button class="subtab-btn active" onclick="showSub(\'exp_first\',this)">',1)
h=h.replace('<div id="sub-exp_first" class="subpanel">','<div id="sub-exp_first" class="subpanel active">',1)
# 4) make Experiment the default-open top-level tab (move active off Methods)
h=h.replace('<div id="tab-methods" class="tab-panel active">','<div id="tab-methods" class="tab-panel">',1)
h=h.replace('<div id="tab-experiment" class="tab-panel">','<div id="tab-experiment" class="tab-panel active">',1)
h=h.replace('<button class="tab-btn active" onclick="showTab(\'methods\',this)">Methods</button>',
            '<button class="tab-btn" onclick="showTab(\'methods\',this)">Methods</button>',1)
h=h.replace('<button class="tab-btn" onclick="showTab(\'experiment\',this)">Experiment</button>',
            '<button class="tab-btn active" onclick="showTab(\'experiment\',this)">Experiment</button>',1)
idx.write_text(h)
print("trimmed. remaining experiment subtab buttons:", re.findall(r"showSub\('([^']+)'", h.split('id="tab-experiment"')[1].split('</div>\n\n<script>')[0]))
print("remaining sub- panels:", re.findall(r'id="(sub-exp_[^"]*|sub-synthlalonde|sub-kallus_[^"]*)"', h))
