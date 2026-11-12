const MESES=["Set","Ago","Jul","Jun","Mai","Abr"];
const PAYOPT=[[-2,"Sem consumo"],[-1,"Pago em dia"],[0,"Uso rotativo"],[1,"1 mês de atraso"],[2,"2 meses"],[3,"3 meses"],[4,"4 meses"],[5,"5 meses"],[6,"6 meses"],[7,"7 meses"],[8,"8+ meses"]];
const $=id=>document.getElementById(id), hist=[];
$("abrir-ajuda").onclick = () => $("ajuda").showModal();

const navLinks = document.querySelectorAll(".nav a");
function updateNavigation() {
  const activeHash = [...navLinks].some(link => link.hash === location.hash) ? location.hash : "#painel";
  for (const link of navLinks) {
    if (link.hash === activeHash) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  }
}
window.addEventListener("hashchange", updateNavigation);
updateNavigation();

const grid=$("months");
grid.insertAdjacentHTML("beforeend",'<div></div>'+MESES.map(m=>`<div class="h">${m}/2005</div>`).join(""));
grid.insertAdjacentHTML("beforeend",'<div class="r">Situação</div>'+MESES.map((m,i)=>`<select id="pay${i}" aria-label="Situação em ${m}">${PAYOPT.map(([v,t])=>`<option value="${v}"${v===0?" selected":""}>${t}</option>`).join("")}</select>`).join(""));
grid.insertAdjacentHTML("beforeend",'<div class="r">Fatura</div>'+MESES.map((m,i)=>`<input id="bill${i}" type="number" value="20000" aria-label="Fatura em ${m}">`).join(""));
grid.insertAdjacentHTML("beforeend",'<div class="r">Pago</div>'+MESES.map((m,i)=>`<input id="paid${i}" type="number" min="0" value="2000" aria-label="Pago em ${m}">`).join(""));

function read(){
  const n=id=>parseFloat($(id).value)||0, r=p=>MESES.map((_,i)=>n(p+i));
  return {limit:n("limit"),age:n("age"),sex:n("sex"),edu:n("edu"),mar:n("mar"),pay:r("pay"),bill:r("bill"),paid:r("paid")};
}

/* Substitua pela chamada ao modelo treinado (ex.: fetch("/predict", ...)).
   Retorna {prob: 0..1, factors: [{name, value}]}; value > 0 aumenta o risco. */
function predict(x){
  const w=[0.55,0.25,0.15,0.1,0.08,0.05], c=v=>Math.max(-1,Math.min(6,v)), f=[];
  f.push({name:"Atrasos no pagamento",value:0.9*x.pay.reduce((s,p,i)=>s+w[i]*c(p),0)});
  const util=x.limit>0?Math.min(1.5,Math.max(0,x.bill[0]/x.limit)):1;
  f.push({name:"Fatura atual vs. limite",value:0.9*(util-0.4)});
  const ratio=x.bill[1]>0?Math.min(1,x.paid[0]/x.bill[1]):1;
  f.push({name:"Parcela paga da fatura anterior",value:-0.9*(ratio-0.15)});
  f.push({name:"Limite de crédito",value:-0.25*Math.log10(Math.max(x.limit,1000)/50000)});
  return {prob:1/(1+Math.exp(-(-1.45+f.reduce((s,a)=>s+a.value,0)))),factors:f};
}

function chart(){
  const x=read(), b=[...x.bill].reverse(), p=[...x.paid].reverse(), m=[...MESES].reverse();
  const max=Math.max(1,...b,...p), H=140, y=v=>150-Math.max(0,v)/max*H;
  let s=[0,.5,1].map(t=>`<line x1="40" x2="550" y1="${150-t*H}" y2="${150-t*H}" stroke="#ececee"/><text x="34" y="${154-t*H}" text-anchor="end">${Math.round(max*t/1000)}k</text>`).join("");
  m.forEach((l,i)=>{const cx=70+i*84;
    s+=`<rect x="${cx-22}" y="${y(b[i])}" width="20" height="${150-y(b[i])}" rx="3" fill="#111"/>`
      +`<rect x="${cx+2}" y="${y(p[i])}" width="20" height="${150-y(p[i])}" rx="3" fill="#c9c9cd"/>`
      +`<text x="${cx}" y="170" text-anchor="middle">${l}</text>`;});
  $("chart").innerHTML=s;
}

let lastProbability;

function renderClassification(prob) {
  const bad = prob >= $("thr").value / 100;
  $("verdict").textContent = bad ? "Provável inadimplente" : "Provável bom pagador";
  $("sub").textContent = bad ? "A probabilidade atingiu o limite escolhido." : "A probabilidade está abaixo do limite escolhido.";
  return bad;
}

function analyze(){
  const x = read();
  const {prob, factors} = predict(x);
  lastProbability = prob;
  const p = Math.round(prob * 100);
  $("pct").textContent = p + "%";
  $("arc").setAttribute("stroke-dasharray", `${prob * 100} 100`);
  const bad = renderClassification(prob);
  $("fx").innerHTML=[...factors].sort((a,b)=>Math.abs(b.value)-Math.abs(a.value)).slice(0,3).map(a=>
    `<li><span>${a.name}</span><b>${a.value>0?"↑ aumenta":"↓ reduz"}</b></li>`).join("");
  hist.unshift({t:new Date().toLocaleTimeString("pt-BR",{hour:"2-digit",minute:"2-digit"}),p,bad,l:x.limit}); hist.length=Math.min(hist.length,5);
  $("hist").innerHTML=hist.map(h=>`<li><span>Limite NT$ ${h.l.toLocaleString("pt-BR")}<small>${h.t} · ${h.p}% de risco</small></span><span class="tag${h.bad?" in":""}">${h.bad?"Inadimplente":"Bom pagador"}</span></li>`).join("");
}

const PRESETS={
  good:{limit:200000,age:42,pay:[-1,-1,-1,-1,-1,-1],bill:[12000,11000,9000,10000,8000,9500],paid:[12000,11000,9000,10000,8000,9500]},
  bad:{limit:30000,age:27,pay:[3,3,2,2,2,0],bill:[29500,28000,27000,26000,25000,22000],paid:[0,0,1000,0,500,1500]}
};
document.querySelectorAll("[data-preset]").forEach(b=>b.onclick=()=>{
  const p=PRESETS[b.dataset.preset]; $("limit").value=p.limit; $("age").value=p.age;
  MESES.forEach((_,i)=>{$("pay"+i).value=p.pay[i];$("bill"+i).value=p.bill[i];$("paid"+i).value=p.paid[i];}); chart(); analyze();
});
document.querySelectorAll("input,select").forEach(e=>e.addEventListener("input",()=>{if(e.id!=="thr")chart();}));
$("run").onclick = () => {
  analyze();
  if (window.matchMedia("(max-width: 980px)").matches) location.hash = "#analises";
};
$("thr").oninput=e=>{$("thv").textContent=e.target.value+"%"; if(lastProbability !== undefined) renderClassification(lastProbability);};
chart();
