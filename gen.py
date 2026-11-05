# Narração VetSmart com a voz do vídeo v3 (pt-BR-FranciscaNeural) + transcrição de conferência (Whisper)
import asyncio, json, os
import edge_tts
VOICE, RATE = "pt-BR-FranciscaNeural", "+4%"
F = json.load(open("frases.json", encoding="utf-8"))
async def one(key, txt):
    words = []; com = edge_tts.Communicate(txt, VOICE, rate=RATE, boundary="WordBoundary")
    with open(f"out/{key}.mp3", "wb") as fh:
        async for ch in com.stream():
            if ch["type"] == "audio": fh.write(ch["data"])
            elif ch["type"] == "WordBoundary": words.append({"w": ch["text"], "t": round(ch["offset"] / 1e7, 3), "d": round(ch["duration"] / 1e7, 3)})
    return key, words
async def main():
    os.makedirs("out", exist_ok=True)
    res = {}
    for k, t in F.items():
        key, w = await one(k, t); res[key] = {"text": t, "words": w}; print("ok", key, flush=True)
    json.dump(res, open("out/words.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
asyncio.run(main())
from faster_whisper import WhisperModel
m = WhisperModel("small", device="cpu", compute_type="int8")
tr = {}
for k in F:
    segs, _ = m.transcribe(f"out/{k}.mp3", language="pt", beam_size=5)
    tr[k] = " ".join(s.text.strip() for s in segs); print(k, "->", tr[k], flush=True)
json.dump(tr, open("out/transcricao.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
