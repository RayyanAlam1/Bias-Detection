"""
manual_test.py  —  Interactive manual tester for the bias model.
Run:  python manual_test.py
Then type any news headline/paragraph and press Enter.
Type 'quit' to exit.
Type 'batch' to run 5 known examples automatically.
"""

import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

CKPT = r"C:\Users\22K-2127\Desktop\FYP\roberta-large-finetuned-v3\checkpoint-502"
LABEL = {0: "LEFT  🔵", 1: "CENTER ⚖️", 2: "RIGHT 🔴"}

print("Loading model …")
tokenizer = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSequenceClassification.from_pretrained(CKPT)
model.eval()
print("Model ready.\n")

KNOWN_EXAMPLES = [
    # (expected_label, text) — real verbatim excerpts from balanced_bias_news_dataset.csv

    # LEFT examples
    (0,
     "On Saturday , women and their allies will take to the streets in cities around the world for the third annual Women s March . "
     "The climate crisis of the 21st century has been caused largely by just 90 companies , which between them produced nearly "
     "two-thirds of the greenhouse gas emissions since the dawn of the industrial age , a major new study suggests . "
     "Unsustainable use of resources is wrecking the planet but recycling is falling short of what is needed to fix it . "
     "The wealth premium has collapsed precipitously over the past 50 years . White families with college degrees now hold "
     "less wealth relative to their white noncollege counterparts than they did in the late 1980s ."),

    (0,
     "Story highlights The federal government is headed for a potential shutdown Republicans have tied the key spending bill "
     "to stopping money for Obamacare Large part of Affordable Care Act is set to go into effect on October 1 "
     "Over the next few days , the drama of a potential government shutdown will collide with the promise of a new health "
     "insurance system known as Obamacare . Here are answers to eight of the most pressing questions about both : "
     "1 . What happens on October 1 with Obamacare and the government shutdown ? "
     "First , the health insurance exchanges established by the Affordable Care Act or Obamacare will be open for business . "
     "Millions of uninsured Americans will be able to enroll in health plans before the law kicks in on January 1 , 2014 . "
     "Second , the U.S. government might shut down if lawmakers can not agree to pass a funding bill ."),

    # CENTER examples
    (1,
     "Robert Mueller s Russia Report Is Coming Thursday . Here s What You Need To Know "
     "The Justice Department says it plans to release special counsel Robert Mueller s report on Thursday morning . "
     "Mueller was appointed in the spring of 2017 to investigate whether President Trump s campaign conspired with "
     "the Russian interference in the 2016 election . The fact of the interference itself had been long established , "
     "and last month Attorney General William Barr told Congress that Mueller did not find that Trump s campaign was involved with it . "
     "Barr also told Congress that Mueller didn t establish that Trump broke the law in trying to frustrate the investigation "
     "but neither did Mueller s office exonerate the president . "
     "Trump and Republicans have welcomed Barr s summary , which they say vindicates the president . "
     "Democrats say they can t be sure Barr is not providing political cover for Trump ."),

    (1,
     "MALAGA , Spain Spanish King Juan Carlos always said that he would die wearing his crown but he unexpectedly stepped down "
     "Monday in favor of his son Crown Prince Felipe , Spanish Prime Minister Mariano Rajoy announced in a nationwide television broadcast . "
     "The abdication 39 years after Juan Carlos ascended to the throne , a period when the king oversaw Spain s transition to democracy "
     "in the wake of the nation s notorious dictator Francisco Franco , comes as corruption scandals have dogged the royal family , "
     "and as the monarchy has seen renewed calls for its disbandment . "
     "Prince Felipe , the monarch s only son , is considered to be well-liked . "
     "Even though the debate as to whether he should step down has raged for some time , most Spaniards expressed astonishment at the announcement ."),

    # RIGHT examples
    (2,
     "As we all were last night , President Trump was watching Minneapolis descend into total anarchy . "
     "Our own Julio Rosas is on the ground there , and the city appears to be completely out of control . "
     "The special counsel is going to keep digging until Trump stops this . It s a new witch hunt . "
     "President Trump s decision to recognize Jerusalem as Israel s capital is a perfectly defensible foreign policy choice . "
     "Speaking from one of the states hardest hit by the opioid epidemic , President Trump signed legislation to address the crisis ."),

    (2,
     "Sen. Elizabeth Warren is by far the most popular would-be presidential candidate among progressives in the Democratic Party . "
     "With a hearty handshake and broad smiles , President Trump and Russian President Vladimir Putin opened their closely watched summit . "
     "The Supreme Court rejected an ACLU-backed bid to slow deportations Thursday , dealing a blow to immigration advocates . "
     "Hillary Clinton has the high-profile backers , the lead in delegates and the well-oiled political machine . "
     "The special counsel is going to keep digging until Trump stops this . It s a new witch hunt . "
     "Authorities in Iowa have captured the suspected gunman in the ambush-style attack on two police officers ."),
]

def predict(text):
    inputs = tokenizer(text.strip(), return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred  = int(torch.argmax(logits).item())
    return pred, probs

def run_batch():
    print("\n" + "="*70)
    print("BATCH TEST — 6 known examples")
    print("="*70)
    correct = 0
    for expected, text in KNOWN_EXAMPLES:
        pred, probs = predict(text)
        status = "PASS" if pred == expected else "FAIL"
        if pred == expected:
            correct += 1
        short = text[:65] + "…" if len(text) > 65 else text
        print(f"[{status}] Expected={LABEL[expected]}  Got={LABEL[pred]}")
        print(f"       L={probs[0]*100:5.1f}%  C={probs[1]*100:5.1f}%  R={probs[2]*100:5.1f}%")
        print(f"       \"{short}\"")
        print()
    print(f"Score: {correct}/{len(KNOWN_EXAMPLES)}")
    print("="*70 + "\n")

# Auto-run batch on start
run_batch()

print("Type any text to test, 'batch' to re-run examples, or 'quit' to exit.\n")
while True:
    try:
        text = input(">>> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not text:
        continue
    if text.lower() == "quit":
        break
    if text.lower() == "batch":
        run_batch()
        continue

    pred, probs = predict(text)
    print(f"\nPrediction : {LABEL[pred]}")
    print(f"Confidence : L={probs[0]*100:5.1f}%  C={probs[1]*100:5.1f}%  R={probs[2]*100:5.1f}%\n")
