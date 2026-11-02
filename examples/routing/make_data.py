"""A harder bilingual routing benchmark for ablations.

    python examples/routing/make_data.py --out examples/routing/data

Intents are (action, object) pairs, 8 x 8 = 64. Messages paraphrase both parts
(``I'd like to end my monthly plan`` / ``我想停掉包月计划``) while the options are
always the plain English intent names (``cancel subscription``), so the model
has to learn the mapping rather than copy words.

16 of the 64 intents never appear in train/dev, not even as distractors. ``test_unseen``
asks only about those, so a model can only get them right by reading the
option text compositionally.

Questions per message:
  * intent   (choice) 3-8 options, half of the distractors share the action or object
  * urgency  (score)  1..5
  * mentions (bool)   "does the message mention <object>?" with a varying object
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ACTIONS = {
    "cancel": {"en": ["I want to stop {o}", "please terminate {o}", "I'd like to end {o}"],
               "zh": ["我想停掉{o}", "请帮我终止{o}", "{o}我不想要了"]},
    "upgrade": {"en": ["I want a bigger version of {o}", "move {o} to a higher tier", "can I get a better {o}"],
                "zh": ["我想把{o}换个更高级的", "帮我提升一下{o}", "{o}能不能换成更好的"]},
    "refund": {"en": ["give me my money back for {o}", "I want to be reimbursed for {o}", "return the payment for {o}"],
               "zh": ["{o}的钱退给我", "我要求退还{o}的费用", "请返还{o}的款项"]},
    "track": {"en": ["where is {o} right now", "what is the status of {o}", "no update on {o} yet"],
              "zh": ["{o}现在到哪一步了", "我想查一下{o}的进度", "{o}一直没有更新"]},
    "change": {"en": ["I need to update {o}", "please modify {o}", "I want to edit {o}"],
               "zh": ["我要更新{o}", "请修改一下{o}", "我想改{o}"]},
    "report problem with": {"en": ["something is broken with {o}", "there is an error in {o}", "I have a problem with {o}"],
                            "zh": ["{o}出故障了", "{o}报错了", "我的{o}有问题"]},
    "ask about": {"en": ["I have a question about {o}", "can you explain {o}", "tell me more about {o}"],
                  "zh": ["我想咨询一下{o}", "能解释一下{o}吗", "想了解{o}"]},
    "renew": {"en": ["I want to extend {o}", "please continue {o}", "keep {o} going"],
              "zh": ["我想延长{o}", "请继续保留{o}", "帮我把{o}续上"]},
}
OBJECTS = {
    "subscription": {"en": ["my membership plan", "my monthly plan", "the recurring plan"],
                     "zh": ["会员套餐", "包月计划", "自动续费套餐"], "name_zh": "订阅"},
    "order": {"en": ["my purchase", "the stuff I bought", "my recent buy"],
              "zh": ["我买的东西", "这笔购物", "最近的购买"], "name_zh": "订单"},
    "password": {"en": ["my login secret", "my sign-in code", "the passcode"],
                 "zh": ["登录口令", "登录密钥", "开机口令"], "name_zh": "密码"},
    "address": {"en": ["where I live", "my home location", "my mailing place"],
                "zh": ["收货地点", "住址", "寄送位置"], "name_zh": "地址"},
    "invoice": {"en": ["the bill", "my receipt", "the billing statement"],
                "zh": ["账单", "收据", "开票信息"], "name_zh": "发票"},
    "delivery": {"en": ["the shipment", "my parcel", "the courier package"],
                 "zh": ["包裹", "快递", "物流"], "name_zh": "配送"},
    "account": {"en": ["my profile", "my user page", "the login profile"],
                "zh": ["个人资料", "用户主页", "账户信息"], "name_zh": "账户"},
    "device": {"en": ["my phone", "the gadget", "my hardware"],
               "zh": ["手机", "设备", "硬件"], "name_zh": "设备"},
}
URGENCY = {
    1: {"en": ["no rush at all.", "whenever you have time."], "zh": ["完全不着急。", "有空再处理就行。"]},
    2: {"en": ["please look at it this week.", "would be nice to sort out soon."], "zh": ["这周内处理就好。", "希望尽快弄好。"]},
    3: {"en": ["please help.", "I need this sorted."], "zh": ["请帮忙处理。", "需要解决一下。"]},
    4: {"en": ["this is urgent, please reply today.", "it is blocking my work."], "zh": ["比较急，请今天回复。", "已经影响我工作了。"]},
    5: {"en": ["URGENT!!! fix this immediately!", "emergency, I need this right now!"], "zh": ["非常紧急！！！马上处理！", "紧急情况，现在就要！"]},
}
GREET = {"en": ["Hi,", "Hello team,", "", "Hey,"], "zh": ["你好，", "客服你好，", "", "在吗？"]}
Q = {
    "en": {"intent": "What does the customer want?", "urgency": "How urgent is this message?",
           "mentions": "Does the message mention the {x}?"},
    "zh": {"intent": "客户想做什么？", "urgency": "这条消息有多紧急？", "mentions": "消息里是否提到了{x}？"},
}

A_KEYS, O_KEYS = list(ACTIONS), list(OBJECTS)
HELD_OUT = {(A_KEYS[i], O_KEYS[(i + s) % 8]) for i in range(8) for s in (0, 3)}  # each action / object twice
ALL = [(a, o) for a in A_KEYS for o in O_KEYS]
SEEN = [x for x in ALL if x not in HELD_OUT]


def intent_name(a: str, o: str) -> str:
    return f"{a} {o}"


def make(rng: random.Random, i: int, pool: list[tuple[str, str]], option_pool: list[tuple[str, str]]) -> dict:
    lang = rng.choice(["en", "zh"])
    a, o = rng.choice(pool)
    urgency = rng.choices([1, 2, 3, 4, 5], weights=[2, 3, 4, 3, 2])[0]
    body = rng.choice(ACTIONS[a][lang]).format(o=rng.choice(OBJECTS[o][lang]))
    sep = " " if lang == "en" else ""
    end = "." if lang == "en" else "。"
    msg = sep.join(p for p in [rng.choice(GREET[lang]), body + end, rng.choice(URGENCY[urgency][lang])] if p)

    k = rng.randint(3, 8)
    hard = [x for x in option_pool if x != (a, o) and (x[0] == a or x[1] == o)]
    rest = [x for x in option_pool if x != (a, o) and x not in hard]
    n_hard = min(len(hard), (k - 1 + 1) // 2)
    distractors = rng.sample(hard, n_hard) + rng.sample(rest, k - 1 - n_hard)
    options = [intent_name(*x) for x in distractors + [(a, o)]]
    rng.shuffle(options)

    target_obj = o if rng.random() < 0.5 else rng.choice([x for x in O_KEYS if x != o])
    obj_name = target_obj if lang == "en" else OBJECTS[target_obj]["name_zh"]
    q = Q[lang]
    return {
        "id": f"r{i:05d}",
        "state": msg,
        "questions": {
            "intent": {"type": "choice", "text": q["intent"], "options": options, "label": intent_name(a, o)},
            "urgency": {"type": "score", "text": q["urgency"], "min": 1, "max": 5, "label": urgency},
            "mentions": {"type": "bool", "text": q["mentions"].format(x=obj_name), "label": target_obj == o},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--dev", type=int, default=500)
    ap.add_argument("--test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    i = 0
    # held-out intents never appear in train/dev, not even as distractors
    for split, n, pool, opts in (("train", a.train, SEEN, SEEN), ("dev", a.dev, SEEN, SEEN),
                                 ("test_seen", a.test, SEEN, SEEN), ("test_unseen", a.test, sorted(HELD_OUT), ALL)):
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for _ in range(n):
                f.write(json.dumps(make(rng, i, pool, opts), ensure_ascii=False) + "\n")
                i += 1
        print(f"wrote {n} examples to {out / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()
