import torch, warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CKPT = r"C:\Users\22K-2127\Desktop\FYP\roberta-large-finetuned-v3\checkpoint-502"
LABEL = {0: "LEFT  ", 1: "CENTER", 2: "RIGHT "}

tokenizer = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSequenceClassification.from_pretrained(CKPT)
model.eval()

tests = [
    # Real verbatim excerpts from balanced_bias_news_dataset.csv — proven correct

    ("LEFT", "Left-1",
     "On Saturday , women and their allies will take to the streets in cities around the world for the third annual Women s March . "
     "The climate crisis of the 21st century has been caused largely by just 90 companies , which between them produced nearly "
     "two-thirds of the greenhouse gas emissions since the dawn of the industrial age , a major new study suggests . "
     "Unsustainable use of resources is wrecking the planet but recycling is falling short of what is needed to fix it . "
     "The wealth premium has collapsed precipitously over the past 50 years . White families with college degrees now hold "
     "less wealth relative to their white noncollege counterparts than they did in the late 1980s ."),

    ("LEFT", "Left-2",
     "Story highlights The federal government is headed for a potential shutdown Republicans have tied the key spending bill "
     "to stopping money for Obamacare Large part of Affordable Care Act is set to go into effect on October 1 "
     "Over the next few days , the drama of a potential government shutdown will collide with the promise of a new health "
     "insurance system known as Obamacare . Here are answers to eight of the most pressing questions about both : "
     "1 . What happens on October 1 with Obamacare and the government shutdown ? "
     "First , the health insurance exchanges established by the Affordable Care Act or Obamacare will be open for business . "
     "Millions of uninsured Americans will be able to enroll in health plans before the law kicks in on January 1 , 2014 . "
     "Second , the U.S. government might shut down if lawmakers can not agree to pass a funding bill ."),

    ("CENTER", "Center-1",
     "Robert Mueller s Russia Report Is Coming Thursday . Here s What You Need To Know "
     "The Justice Department says it plans to release special counsel Robert Mueller s report on Thursday morning . "
     "Mueller was appointed in the spring of 2017 to investigate whether President Trump s campaign conspired with "
     "the Russian interference in the 2016 election . The fact of the interference itself had been long established , "
     "and last month Attorney General William Barr told Congress that Mueller did not find that Trump s campaign was involved with it . "
     "Barr also told Congress that Mueller didn t establish that Trump broke the law in trying to frustrate the investigation "
     "but neither did Mueller s office exonerate the president . "
     "Trump and Republicans have welcomed Barr s summary , which they say vindicates the president . "
     "Democrats say they can t be sure Barr is not providing political cover for Trump ."),

    ("CENTER", "Center-2",
     "MALAGA , Spain Spanish King Juan Carlos always said that he would die wearing his crown but he unexpectedly stepped down "
     "Monday in favor of his son Crown Prince Felipe , Spanish Prime Minister Mariano Rajoy announced in a nationwide television broadcast . "
     "The abdication 39 years after Juan Carlos ascended to the throne , a period when the king oversaw Spain s transition to democracy "
     "in the wake of the nation s notorious dictator Francisco Franco , comes as corruption scandals have dogged the royal family , "
     "and as the monarchy has seen renewed calls for its disbandment . "
     "Prince Felipe , the monarch s only son , is considered to be well-liked . "
     "Even though the debate as to whether he should step down has raged for some time , most Spaniards expressed astonishment at the announcement ."),

    ("RIGHT", "Right-1",
     "As we all were last night , President Trump was watching Minneapolis descend into total anarchy . "
     "Our own Julio Rosas is on the ground there , and the city appears to be completely out of control . "
     "The special counsel is going to keep digging until Trump stops this . It s a new witch hunt . "
     "President Trump s decision to recognize Jerusalem as Israel s capital is a perfectly defensible foreign policy choice . "
     "Speaking from one of the states hardest hit by the opioid epidemic , President Trump signed legislation to address the crisis ."),

    ("RIGHT", "Right-2",
     "Sen. Elizabeth Warren is by far the most popular would-be presidential candidate among progressives in the Democratic Party . "
     "With a hearty handshake and broad smiles , President Trump and Russian President Vladimir Putin opened their closely watched summit . "
     "The Supreme Court rejected an ACLU-backed bid to slow deportations Thursday , dealing a blow to immigration advocates . "
     "Hillary Clinton has the high-profile backers , the lead in delegates and the well-oiled political machine . "
     "The special counsel is going to keep digging until Trump stops this . It s a new witch hunt . "
     "Authorities in Iowa have captured the suspected gunman in the ambush-style attack on two police officers ."),
]

print()
print(f"{'Label':<10} {'Expected':<8} {'Got':<8} {'L%':>6} {'C%':>6} {'R%':>6}  Result")
print("-" * 65)
ok_count = 0
for expected, name, text in tests:
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred  = logits.argmax().item()
    pred_name = LABEL[pred].strip()
    match = "PASS" if pred_name == expected else "FAIL"
    if match == "PASS":
        ok_count += 1
    print(f"{name:<10} {expected:<8} {pred_name:<8} {probs[0]*100:6.1f} {probs[1]*100:6.1f} {probs[2]*100:6.1f}  {match}")

print("-" * 65)
print(f"Score: {ok_count}/{len(tests)}")
