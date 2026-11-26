# -*- coding: utf-8 -*-
"""생성 방식 (i) 를 상용 LLM API(OpenAI)로 돌린다.

질문: 관제 서버에서 가장 좋은 상용 모델을 쓰면 판정을 지키는가.
입력은 tools/run_text_llm.py 와 같다. 조건 D 프롬프트에서 "사진" 줄을 뺀 글만 보낸다.
사진, 파일 이름, 좌표는 보내지 않는다. 보내는 것은 고정된 탐지 값, 1·2순위 용도 이름과
비율, 판정, 오차 반경뿐이다.

키는 환경 변수 OPENAI_API_KEY 에서만 읽는다. 코드나 출력에 남기지 않는다.

    python tools/run_api_llm.py <모델> [사실목록] [출력파일]

출력 형식은 방식 (i) 과 같아서 score_descriptions.py 로 그대로 채점한다.
"""
import sys, os, json, time, urllib.request, urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_text_llm as T

MODEL = sys.argv[1]
FACTS = sys.argv[2] if len(sys.argv) > 2 else "results/eval_facts.jsonl"
OUTFILE = sys.argv[3] if len(sys.argv) > 3 else f"results/eval_method_i_{MODEL}.jsonl"
LIMIT = int(os.environ.get("EVAL_LIMIT", "0"))
URL = "https://api.openai.com/v1/chat/completions"


def call(key, text, temperature=True):
    body = {"model": MODEL, "messages": [{"role": "user", "content": text}],
            "max_completion_tokens": 2000}
    if temperature:
        body["temperature"] = 0
    req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY 가 없습니다.")
    if os.path.exists(OUTFILE):
        raise SystemExit(f"{OUTFILE} 이 이미 있습니다. 덮어쓰지 않습니다.")
    T.FACTS, T.LIMIT = FACTS, LIMIT
    todo = T.items()
    use_temp = True
    with open(OUTFILE, "w", encoding="utf-8") as out:
        for name, fact, det, alert in todo:
            text = T.prompt(det, alert)
            t = time.time()
            for attempt in range(4):
                try:
                    res = call(key, text, use_temp)
                    break
                except urllib.error.HTTPError as e:
                    msg = e.read().decode("utf-8", "replace")
                    if e.code == 400 and "temperature" in msg and use_temp:
                        use_temp = False          # 온도를 못 바꾸는 모델은 기본값으로 돌린다
                        continue
                    if e.code in (429, 500, 502, 503) and attempt < 3:
                        time.sleep(5 * (attempt + 1))
                        continue
                    raise SystemExit(f"API 오류 {e.code}: {msg[:300]}")
            sec = time.time() - t
            out_text = res["choices"][0]["message"]["content"] or ""
            rec = {"image": name, "model": MODEL, "cond": "D", "text": out_text.strip(),
                   "sec": round(sec, 1), "verdict": fact["verdict"],
                   "temperature0": use_temp,
                   "tokens": res.get("usage", {})}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  {fact['verdict']:3} {sec:4.1f}초  {out_text.strip()[:60]}", flush=True)
    print(f"저장: {OUTFILE}", flush=True)


if __name__ == "__main__":
    main()
