"""Generate a small bilingual (EN / 中文) support-ticket dataset with three question types.

    python examples/toy_tickets/make_data.py --out examples/toy_tickets/data

Each ticket gets:
  * team     (choice) billing / shipping / technical / account
  * urgency  (score)  1..5
  * refund   (bool)   does the customer ask for a refund?
It is synthetic and template-based: good for smoke tests and for showing the
data format, not for measuring real-world quality.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TOPICS = {
    "billing": {
        "en": ["I was charged twice for my subscription", "my invoice shows the wrong amount",
               "the payment failed but money left my card", "why did my plan price go up",
               "I need a copy of last month's invoice"],
        "zh": ["我的订阅被重复扣费了", "发票上的金额不对", "付款失败了但卡里的钱被扣了",
               "为什么套餐价格涨了", "我需要上个月的发票"],
    },
    "shipping": {
        "en": ["my package has not arrived yet", "the tracking number does not work",
               "the parcel was delivered to the wrong address", "the box arrived damaged",
               "can I change the delivery date"],
        "zh": ["我的包裹还没到", "物流单号查不到", "快递送错地址了", "收到的箱子是破的", "能改一下送货日期吗"],
    },
    "technical": {
        "en": ["the app crashes when I open settings", "I get error 500 on the dashboard",
               "the export button does nothing", "sync stopped working after the update",
               "the page loads forever"],
        "zh": ["打开设置时应用闪退", "仪表盘报 500 错误", "点导出按钮没反应", "更新后同步不工作了", "页面一直在加载"],
    },
    "account": {
        "en": ["I cannot log in to my account", "please reset my password",
               "how do I change the email on my account", "my account was locked",
               "I want to delete my account"],
        "zh": ["我登录不了账号", "请帮我重置密码", "怎么修改账号绑定的邮箱", "我的账号被锁了", "我想注销账号"],
    },
}
URGENCY = {
    1: {"en": ["no rush, whenever you have time.", "just curious."], "zh": ["不着急，有空再看。", "随便问问。"]},
    2: {"en": ["please look at it this week.", "would be nice to fix soon."], "zh": ["这周内处理就行。", "希望尽快修一下。"]},
    3: {"en": ["please help.", "I need this fixed."], "zh": ["请帮忙处理。", "需要解决一下。"]},
    4: {"en": ["this is urgent, please reply today.", "it is blocking my work."], "zh": ["比较急，请今天回复。", "已经影响我工作了。"]},
    5: {"en": ["URGENT!!! fix this immediately, customers are waiting!", "emergency, everything is down right now!"],
        "zh": ["非常紧急！！！请立刻处理，客户都在等！", "紧急情况，现在全部瘫痪了！"]},
}
REFUND = {"en": ["I want a refund.", "please refund my money."], "zh": ["我要求退款。", "请把钱退给我。"]}
NOISE = {"en": ["Hi team,", "Hello,", "", "Thanks."], "zh": ["你好，", "客服你好，", "", "谢谢。"]}
CHANNELS = ["email", "chat", "app"]

QUESTIONS = {
    "en": {
        "team": "Which team should handle this ticket?",
        "urgency": "How urgent is this ticket?",
        "refund": "Is the customer asking for a refund?",
    },
    "zh": {
        "team": "这张工单应该交给哪个团队？",
        "urgency": "这张工单有多紧急？",
        "refund": "客户是否要求退款？",
    },
}


def make_example(rng: random.Random, i: int) -> dict:
    lang = rng.choice(["en", "zh"])
    team = rng.choice(list(TOPICS))
    urgency = rng.choices([1, 2, 3, 4, 5], weights=[2, 3, 4, 3, 2])[0]
    refund = team in ("billing", "shipping") and rng.random() < 0.45
    sep = " " if lang == "en" else ""
    parts = [rng.choice(NOISE[lang]), rng.choice(TOPICS[team][lang]) + ("," if lang == "en" else "，"),
             rng.choice(URGENCY[urgency][lang])]
    if refund:
        parts.append(rng.choice(REFUND[lang]))
    text = sep.join(p for p in parts if p).strip()
    q = QUESTIONS[lang]
    options = list(TOPICS)
    rng.shuffle(options)
    return {
        "id": f"t{i:05d}",
        "state": {"channel": rng.choice(CHANNELS), "lang": lang, "message": text},
        "questions": {
            "team": {"type": "choice", "text": q["team"], "options": options, "label": team},
            "urgency": {"type": "score", "text": q["urgency"], "min": 1, "max": 5, "label": urgency},
            "refund": {"type": "bool", "text": q["refund"], "label": refund},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--dev", type=int, default=300)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    i = 0
    for split, n in (("train", a.train), ("dev", a.dev), ("test", a.test)):
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for _ in range(n):
                f.write(json.dumps(make_example(rng, i), ensure_ascii=False) + "\n")
                i += 1
        print(f"wrote {n} examples to {out / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()
