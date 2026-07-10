var CHART_REG={};
function toggleDD(btn){var menu=btn.nextElementSibling,willOpen=!menu.classList.contains('open');
  document.querySelectorAll('.chart-dd-menu.open').forEach(function(m){m.classList.remove('open');});
  if(willOpen)menu.classList.add('open');}
document.addEventListener('click',function(e){if(!e.target.closest('.chart-dd'))document.querySelectorAll('.chart-dd-menu.open').forEach(function(m){m.classList.remove('open');});});
function setSeriesVis(id,mid,on){var reg=CHART_REG[id];if(reg)reg.hidden[mid]=!on;
  document.querySelectorAll('#'+id+' g[data-method="'+mid+'"]').forEach(function(g){g.style.display=on?'':'none';});
  var li=document.querySelector('#'+id+' .pchart-leg-item[data-method="'+mid+'"]');if(li)li.classList.toggle('off',!on);
  var cb=document.querySelector('#'+id+' .chart-dd-item input[data-method="'+mid+'"]');if(cb)cb.checked=on;}
function toggleSeries(id,mid,on){setSeriesVis(id,mid,on);}
function toggleLeg(id,mid){var reg=CHART_REG[id];setSeriesVis(id,mid,reg&&reg.hidden[mid]?true:false);}
function _cfmt(v){return Math.abs(v)>=10?v.toFixed(0):(Math.round(v*1000)/1000);}
/* ---- method styling: COLOR by estimator family, SHAPE by uncertainty set ----
   family (1st slot) -> colour; suffix -> line/marker:  -X-X dashed line (no marker),
   -O-X solid + square,  -O-W solid + circle.  Kallus solid + triangle, Oracle dotted. */
var FAMILY_COLOR={IPW:'#1f77b4',Hajek:'#9467bd',DoublyRobust:'#2ca02c',Direct:'#ff7f0e',Kallus:'#d62728',Oracle:'#444444'};
function _cmstyle(name,fb){var n=String(name||'').split('(')[0].trim();var fam=n.split('-')[0];
  var color=FAMILY_COLOR[fam]||fb||'#888',dash=null,marker=null;
  if(/-O-W$/.test(n))marker='circle';
  else if(/-O-X$/.test(n))marker='square';
  else if(/-X-X$/.test(n))dash='6,4';
  else if(fam==='Kallus')marker='triangle';
  else if(fam==='Oracle')dash='2,3';
  return {color:color,dash:dash,marker:marker};}
function _cmk(marker,cx,cy,col){cx=parseFloat(cx);cy=parseFloat(cy);
  if(marker==='square')return '<rect x="'+(cx-3)+'" y="'+(cy-3)+'" width="6" height="6" fill="'+col+'"/>';
  if(marker==='triangle')return '<polygon points="'+cx+','+(cy-3.8)+' '+(cx+3.5)+','+(cy+2.9)+' '+(cx-3.5)+','+(cy+2.9)+'" fill="'+col+'"/>';
  return '<circle cx="'+cx+'" cy="'+cy+'" r="3.1" fill="'+col+'"/>';}
function _mswatch(st){var ln='<line x1="2" y1="6" x2="24" y2="6" stroke="'+st.color+'" stroke-width="2.4"'+(st.dash?' stroke-dasharray="'+st.dash+'"':'')+'/>';
  return '<svg class="msw" width="26" height="12" viewBox="0 0 26 12" style="vertical-align:middle">'+ln+(st.marker?_cmk(st.marker,13,6,st.color):'')+'</svg>';}
/* effective per-series style: family default from _cmstyle, overridden by explicit se.color/marker/dash/width */
function _seriesStyle(se){var st=_cmstyle(se.label||se.id, se.color);
  /* family colour wins for recognised methods; se.color only applies to unknown labels (via _cmstyle fallback) */
  if('marker' in se)st.marker=se.marker;
  if('dash' in se)st.dash=se.dash;
  st.width=se.width||2.3;return st;}
function _csvg(data,hidden){
  var W=820,H=440,mL=60,mR=18,mT=16,mB=46,x0=mL,x1=W-mR,y0=H-mB,y1=mT;
  var xmin=(data.xmin!=null?data.xmin:Math.min.apply(null,data.x)),xmax=(data.xmax!=null?data.xmax:Math.max.apply(null,data.x));
  var ymin=(data.ymin!=null?data.ymin:0),ymax=(data.ymax!=null?data.ymax:1);
  function sx(x){return (x0+(x-xmin)/(xmax-xmin)*(x1-x0)).toFixed(1);}
  function sy(v){return (y0-(v-ymin)/(ymax-ymin)*(y0-y1)).toFixed(1);}
  var s='<svg viewBox="0 0 '+W+' '+H+'" class="pchart-svg" preserveAspectRatio="xMidYMid meet" role="img">';
  var i,yv,y,px,xv;
  for(i=0;i<5;i++){yv=ymin+(ymax-ymin)*i/4;y=sy(yv);
    s+='<line x1="'+x0+'" y1="'+y+'" x2="'+x1+'" y2="'+y+'" stroke="#eceef4"/>';
    s+='<text x="'+(x0-9)+'" y="'+(parseFloat(y)+4)+'" text-anchor="end" class="pchart-tick">'+_cfmt(yv)+'</text>';}
  if(data.xticks){data.xticks.forEach(function(t){px=sx(t.x);
    s+='<line x1="'+px+'" y1="'+y1+'" x2="'+px+'" y2="'+y0+'" stroke="#f4f5f9"/>';
    s+='<text x="'+px+'" y="'+(y0+22)+'" text-anchor="middle" class="pchart-tick">'+t.label+'</text>';});}
  else{for(i=0;i<5;i++){xv=xmin+(xmax-xmin)*i/4;px=sx(xv);
    s+='<line x1="'+px+'" y1="'+y1+'" x2="'+px+'" y2="'+y0+'" stroke="#f4f5f9"/>';
    s+='<text x="'+px+'" y="'+(y0+22)+'" text-anchor="middle" class="pchart-tick">'+(Math.round(xv*100)/100)+'</text>';}}
  if(data.gammaLine!=null){var gx=sx(data.gammaLine);s+='<line x1="'+gx+'" y1="'+y1+'" x2="'+gx+'" y2="'+y0+'" stroke="#c9ccd6" stroke-dasharray="3,3"/>';}
  (data.hlines||[]).forEach(function(h){if(h.y<ymin||h.y>ymax)return;var hy=sy(h.y);
    s+='<line x1="'+x0+'" y1="'+hy+'" x2="'+x1+'" y2="'+hy+'" stroke="'+(h.color||'#888')+'" stroke-width="1.3" stroke-dasharray="'+(h.dash||'5,4')+'"/>';
    s+='<text x="'+(x1-3)+'" y="'+(parseFloat(hy)-3)+'" text-anchor="end" class="pchart-tick" fill="'+(h.color||'#888')+'">'+h.label+'</text>';});
  s+='<text x="'+((x0+x1)/2)+'" y="'+(H-7)+'" text-anchor="middle" class="pchart-axlab">'+(data.xlabel||'X')+'</text>';
  s+='<text transform="translate(15,'+((y0+y1)/2)+') rotate(-90)" text-anchor="middle" class="pchart-axlab">'+(data.ylabel||'y')+'</text>';
  data.series.forEach(function(se){if(!se.y)return;var hid=hidden&&hidden[se.id];
    var st=_seriesStyle(se);
    s+='<g data-method="'+se.id+'" class="pchart-series" style="display:'+(hid?'none':'')+'">';
    if(se.ysd){var up=[],dn=[];se.y.forEach(function(v,k){if(v==null||se.ysd[k]==null)return;
        up.push(sx(data.x[k])+','+sy(Math.min(ymax,v+se.ysd[k])));dn.push(sx(data.x[k])+','+sy(Math.max(ymin,v-se.ysd[k])));});
      if(up.length>1)s+='<polygon points="'+up.concat(dn.reverse()).join(' ')+'" fill="'+st.color+'" fill-opacity="0.13" stroke="none"/>';}
    var pts=se.y.map(function(v,k){return v==null?null:sx(data.x[k])+','+sy(v);}).filter(function(p){return p!=null;}).join(' ');
    if(pts)s+='<polyline points="'+pts+'" fill="none" stroke="'+st.color+'" stroke-width="'+st.width+'"'+(st.dash?' stroke-dasharray="'+st.dash+'"':'')+' stroke-linejoin="round" stroke-linecap="round"/>';
    if(st.marker)se.y.forEach(function(v,k){if(v!=null)s+=_cmk(st.marker,sx(data.x[k]),sy(v),st.color);});
    s+='</g>';});
  s+='</svg>';return s;}
function setChartGamma(id,gs,btn){var reg=CHART_REG[id];if(!reg||!reg.data.seriesByGamma[gs])return;
  reg.data.series=reg.data.seriesByGamma[gs];reg.data._gamma=gs;
  document.querySelector('#'+id+' .pchart-wrap').innerHTML=_csvg(reg.data,reg.hidden);
  var b=document.querySelector('#'+id+' .gamma-dd-btn span');if(b)b.textContent='Γ = '+gs+' ';
  document.querySelectorAll('#'+id+' .gamma-dd-menu .chart-dd-item').forEach(function(it){it.classList.remove('sel');});
  if(btn)btn.classList.add('sel');
  document.querySelectorAll('.chart-dd-menu.open').forEach(function(m){m.classList.remove('open');});}
function setChartSeed(id,sk,btn){var reg=CHART_REG[id];if(!reg||!reg.data.seriesBySeedGamma||!reg.data.seriesBySeedGamma[sk])return;
  reg.data._seed=sk;reg.data.seriesByGamma=reg.data.seriesBySeedGamma[sk];
  var g=reg.data._gamma;if(!reg.data.seriesByGamma[g])g=Object.keys(reg.data.seriesByGamma)[0];
  reg.data._gamma=g;reg.data.series=reg.data.seriesByGamma[g];
  document.querySelector('#'+id+' .pchart-wrap').innerHTML=_csvg(reg.data,reg.hidden);
  var cl='';reg.data.seeds.forEach(function(s){if(s.key===sk)cl=s.label;});
  var b=document.querySelector('#'+id+' .seed-dd-btn span');if(b)b.textContent=cl+' ';
  document.querySelectorAll('#'+id+' .seed-dd-menu .chart-dd-item').forEach(function(it){it.classList.remove('sel');});
  if(btn)btn.classList.add('sel');
  document.querySelectorAll('.chart-dd-menu.open').forEach(function(m){m.classList.remove('open');});}
function renderChart(id,data){var el=document.getElementById(id);if(!el)return;
  CHART_REG[id]={data:data,hidden:{}};
  if(data.seriesBySeedGamma){var s0=data.defaultSeed!=null?String(data.defaultSeed):data.seeds[data.seeds.length-1].key;
    if(!data.seriesBySeedGamma[s0])s0=data.seeds[0].key;data._seed=s0;data.seriesByGamma=data.seriesBySeedGamma[s0];}
  if(data.gammas&&data.seriesByGamma){var g0=String(data.defaultGamma!=null?data.defaultGamma:data.gammas[data.gammas.length-1]);
    if(!data.seriesByGamma[g0])g0=String(data.gammas[0]);data.series=data.seriesByGamma[g0];data._gamma=g0;}
  var svg=_csvg(data,{});
  var menu='';data.series.forEach(function(se){
    menu+='<label class="chart-dd-item"><input type="checkbox" data-method="'+se.id+'" checked onchange="toggleSeries(\''+id+'\',\''+se.id+'\',this.checked)">'+_mswatch(_seriesStyle(se))+se.label+'</label>';});
  var tools='<div class="chart-dd"><button class="chart-dd-btn" onclick="toggleDD(this)">Show methods ▾</button><div class="chart-dd-menu">'+menu+'</div></div>';
  if(data.gammas&&data.seriesByGamma){var gmenu='';data.gammas.forEach(function(g){var gss=String(g);
      gmenu+='<div class="chart-dd-item'+(gss===data._gamma?' sel':'')+'" onclick="setChartGamma(\''+id+'\',\''+gss+'\',this)">Γ = '+gss+(g===data.matchedGamma?'  (matched)':'')+'</div>';});
    tools+='<div class="chart-dd"><button class="chart-dd-btn gamma-dd-btn" onclick="toggleDD(this)"><span>Γ = '+data._gamma+' </span>▾</button><div class="chart-dd-menu gamma-dd-menu">'+gmenu+'</div></div>';}
  if(data.seriesBySeedGamma){var curlab='';data.seeds.forEach(function(s){if(s.key===data._seed)curlab=s.label;});
    var smenu='';data.seeds.forEach(function(s){smenu+='<div class="chart-dd-item'+(s.key===data._seed?' sel':'')+'" onclick="setChartSeed(\''+id+'\',\''+s.key+'\',this)">'+s.label+'</div>';});
    tools+='<div class="chart-dd"><button class="chart-dd-btn seed-dd-btn" onclick="toggleDD(this)"><span>'+curlab+' </span>▾</button><div class="chart-dd-menu seed-dd-menu">'+smenu+'</div></div>';}
  var leg='<div class="pchart-legend">';data.series.forEach(function(se){
    leg+='<div class="pchart-leg-item" data-method="'+se.id+'" title="click to toggle" onclick="toggleLeg(\''+id+'\',\''+se.id+'\')">'+_mswatch(_seriesStyle(se))+se.label+'</div>';});
  leg+='</div>';
  el.innerHTML='<div class="ichart-toolbar">'+tools+'<span class="ichart-note">'+(data.note||'')+'  ·  toggle in the legend / dropdown</span></div><div class="pchart-row"><div class="pchart-wrap">'+svg+'</div>'+leg+'</div>';}
/* ---- small-multiples variant: one panel per arm, shared method-toggle + Γ-selector (randomized-policy view) ---- */
function _panelData(D,a,gk){return {x:D.x,xmin:D.xmin,xmax:D.xmax,ymin:0,ymax:1,xlabel:D.xlabel||'X',ylabel:(D.armYlabel?D.armYlabel[a]:'π'),series:D.byGammaArm[gk][a]};}
function setPanelsGamma(id,gk,btn){var reg=CHART_REG[id];if(!reg||!reg.data.byGammaArm[gk])return;var D=reg.data;D._gamma=gk;
  document.querySelectorAll('#'+id+' .pchart-wrap').forEach(function(w){var a=+w.getAttribute('data-arm');w.innerHTML=_csvg(_panelData(D,a,gk),reg.hidden);});
  var b=document.querySelector('#'+id+' .gamma-dd-btn span');if(b)b.textContent='Γ = '+gk+' ';
  document.querySelectorAll('#'+id+' .gamma-dd-menu .chart-dd-item').forEach(function(it){it.classList.remove('sel');});
  if(btn)btn.classList.add('sel');
  document.querySelectorAll('.chart-dd-menu.open').forEach(function(m){m.classList.remove('open');});}
function renderChartPanels(id,D){var el=document.getElementById(id);if(!el)return;
  var hid0={};(D.defaultHidden||[]).forEach(function(m){hid0[m]=true;});
  CHART_REG[id]={data:D,hidden:hid0,panels:true};
  var g0=String(D.defaultGamma!=null?D.defaultGamma:D.gammas[D.gammas.length-1]);if(!D.byGammaArm[g0])g0=String(D.gammas[0]);D._gamma=g0;
  var menu='';D.legend.forEach(function(se){
    menu+='<label class="chart-dd-item"><input type="checkbox" data-method="'+se.id+'"'+(hid0[se.id]?'':' checked')+' onchange="toggleSeries(\''+id+'\',\''+se.id+'\',this.checked)">'+_mswatch(_seriesStyle(se))+se.label+'</label>';});
  var tools='<div class="chart-dd"><button class="chart-dd-btn" onclick="toggleDD(this)">Show methods ▾</button><div class="chart-dd-menu">'+menu+'</div></div>';
  var gmenu='';D.gammas.forEach(function(g){var gss=String(g);
    gmenu+='<div class="chart-dd-item'+(gss===D._gamma?' sel':'')+'" onclick="setPanelsGamma(\''+id+'\',\''+gss+'\',this)">Γ = '+gss+(g===D.matchedGamma?'  (matched)':'')+'</div>';});
  tools+='<div class="chart-dd"><button class="chart-dd-btn gamma-dd-btn" onclick="toggleDD(this)"><span>Γ = '+D._gamma+' </span>▾</button><div class="chart-dd-menu gamma-dd-menu">'+gmenu+'</div></div>';
  var panels='';for(var a=0;a<D.arms.length;a++){
    panels+='<div class="ppanel"><div class="ppanel-title">'+D.armTitles[a]+'</div><div class="pchart-wrap" data-arm="'+a+'">'+_csvg(_panelData(D,a,g0),hid0)+'</div></div>';}
  var leg='<div class="pchart-legend">';D.legend.forEach(function(se){
    leg+='<div class="pchart-leg-item'+(hid0[se.id]?' off':'')+'" data-method="'+se.id+'" title="click to toggle" onclick="toggleLeg(\''+id+'\',\''+se.id+'\')">'+_mswatch(_seriesStyle(se))+se.label+'</div>';});
  leg+='</div>';
  el.innerHTML='<div class="ichart-toolbar">'+tools+'<span class="ichart-note">'+(D.note||'')+'  ·  toggle in the legend / dropdown</span></div><div class="pchart-row"><div class="ppanel-grid">'+panels+'</div>'+leg+'</div>';}
