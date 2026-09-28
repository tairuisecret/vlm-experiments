import json, sys, glob, os
r=sys.argv[1]
runs=sorted(glob.glob(f"/mnt/SSD1/gzh/vlm/out_r{r}/v*"))
run=runs[-1]
rows=[json.loads(l) for l in open(f"{run}/logging.jsonl")]
ev=[(x["eval_loss"], x.get("global_step") or x.get("global_step/max_steps")) for x in rows if "eval_loss" in x]
best=min(ev)[1]
if isinstance(best,str): best=best.split("/")[0]
ck=sorted(glob.glob(f"{run}/checkpoint-*"), key=lambda p:int(p.split("-")[-1]))
print(f"{run}/checkpoint-{best}" if os.path.isdir(f"{run}/checkpoint-{best}") else ck[-1], ck[-1], sep="\n")
