const $ = (id)=>document.getElementById(id);
let library = [];
let selected = null;

function now(){return new Date().toLocaleTimeString("fr-FR",{hour:"2-digit",minute:"2-digit",second:"2-digit"});}
$("time1").textContent=now(); $("time2").textContent=now(); $("time3").textContent=now();

async function loadLibrary(){
  try{
    const r=await fetch("data/alpha_dialogue_library.json");
    library=await r.json();
    $("libCount").textContent=library.length;
  }catch(e){ library=[]; }
}
loadLibrary();

function setReply(q, r, t, interpretation, simulated=true){
  $("alphaReply").textContent=r;
  $("translation").textContent=t;
  $("interpretation").textContent=interpretation;
  $("time2").textContent=now(); $("time3").textContent=now();
  $("net").textContent=simulated ? "LOCAL" : "PASSERELLE";
}

function localFallback(question){
  if(!library.length){
    setReply(question,
      "Aucune réponse locale disponible.",
      "La bibliothèque n'est pas chargée.",
      "Ce résultat ne permet aucune conclusion sur Alpha.",
      true);
    return;
  }
  const words=question.toLowerCase().split(/\W+/).filter(Boolean);
  let best=library[0], score=-1;
  for(const item of library){
    const s=words.reduce((n,w)=>n+(item.question.toLowerCase().includes(w)?1:0),0);
    if(s>score){score=s;best=item;}
  }
  setReply(
    question,
    best.fallback_response,
    "Traduction : la réponse de secours indique qu'Alpha se concentre sur le thème identifié, mais qu'aucune conclusion supplémentaire n'est justifiée sans observation.",
    best.interpretation,
    true
  );
}

async function exchange(question){
  const q=question.trim();
  if(!q) return;
  $("alphaReply").textContent="Connexion à la passerelle…";
  $("translation").textContent="En attente de la réponse d'Alpha.";
  $("interpretation").textContent="Aucune interprétation avant réception d'une réponse.";
  $("net").textContent="CONNEXION…";
  try{
    const cfg=await fetch("bridge/config.json").then(r=>r.json());
    if(cfg.enabled && cfg.endpoint){
      const res=await fetch(cfg.endpoint,{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({question:q, mode:"alpha_exchange"})
      });
      if(!res.ok) throw new Error("bridge");
      const data=await res.json();
      setReply(q,data.alpha_response||"Réponse vide.",data.translation||"Aucune traduction fournie.",data.interpretation||"Aucune interprétation fournie.",false);
      return;
    }
  }catch(e){}
  localFallback(q);
}

$("exchange").addEventListener("click",()=>{
  const q=$("question").value.trim() || "Décris ton état actuel et ce sur quoi tu te concentres en ce moment.";
  exchange(q);
});
$("send").addEventListener("click",()=>exchange($("question").value));
$("question").addEventListener("keydown",(e)=>{if((e.ctrlKey||e.metaKey)&&e.key==="Enter") exchange($("question").value);});

$("libraryBtn").addEventListener("click",()=>{
  $("library").classList.remove("hidden"); renderLibrary(library.slice(0,30));
});
$("closeLib").addEventListener("click",()=>$("library").classList.add("hidden"));
$("search").addEventListener("input",()=>{
  const q=$("search").value.toLowerCase();
  renderLibrary(library.filter(x=>x.question.toLowerCase().includes(q)).slice(0,40));
});
function renderLibrary(items){
  $("libraryResults").innerHTML=items.map(x=>`<div class="library-item" data-id="${x.id}">
    <strong>#${x.id} — ${x.question}</strong>
    <small>${x.intent} · ${x.focus}</small>
  </div>`).join("") || "<div class='library-item'>Aucun résultat.</div>";
  document.querySelectorAll(".library-item").forEach(el=>el.addEventListener("click",()=>{
    const item=library.find(x=>x.id==el.dataset.id);
    if(item){$("question").value=item.question;$("library").classList.add("hidden");}
  }));
}

if("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(()=>{});
