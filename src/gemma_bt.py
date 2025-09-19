import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Load Gemma IT model
# -----------------------------
model_name = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -----------------------------
# Data
# -----------------------------
#English monolingual text (press release) 
english_texts = [ """Ministry of Health and Family Welfare On a declining trend, India’s Active Caseload further dips to 23,43,152; Active Cases decrease by 76,755 in last 24 hours At 1.86 Lakh Cases, Daily New Cases are lowest in last 44 days Less than 3 lakh Daily New Cases for 12 consecutive days Recoveries continue to outnumber the Daily New Cases for 15th consecutive day Recovery Rate increases to 90.34% Daily Positivity Rate at 9.00%; less than 10% for 4 consecutive days India’s Active Caseload has now reduced to 23,43,152. Active Cases have decreased since its last peak on 10th May 2021. A net decline of 76,755 is witnessed in the last 24 hours and active cases are now only 8.50%of the country's total Positive Cases. As part of continued decline in the daily new cases, the country has recorded less than 3 lakh Daily New Cases for the twelve consecutive days now. """ ]
# Hindi monolingual text (press release) 
hindi_texts = [ """प्रधानमंत्री कार्यालय प्रधानमंत्री ने गुजरात के मोढेरा में सूर्य मंदिर का दौरा किया प्रधानमंत्री श्री नरेंद्र मोदी ने आज गुजरात के मोढेरा में सूर्य मंदिर का दौरा किया। प्रधानमंत्री के आगमन पर उनका अभिनंदन किया गया। श्री मोदी ने सूर्य मंदिर में हैरिटेज लाइटिंग का उद्घाटन किया। ये भारत का पहला विरासत स्थल बन गया है जो पूरी तरह से सौर ऊर्जा से संचालित है। उन्होंने मोढेरा सूर्य मंदिर के 3डी प्रोजेक्शन मैपिंग का भी उद्घाटन किया। प्रधानमंत्री ने इस मंदिर के इतिहास को दर्शाने वाले एक सांस्कृतिक कार्यक्रम को भी देखा। गुजरात के मुख्यमंत्री श्री भूपेंद्र पटेल, सांसद श्री सी आर पाटिल, गुजरात सरकार के मंत्री श्री पूर्णेशभाई मोदी और श्री अरविंदभाई रैयानी भी प्रधानमंत्री की इस यात्रा पर उनके साथ थे। इससे पहले आज प्रधानमंत्री ने गुजरात में मोढेरा, मेहसाणा में कई परियोजनाओं की आधारशिला रखी और 3900 करोड़ रुपये से अधिक की परियोजनाएं राष्ट्र को समर्पित की। प्रधानमंत्री ने मोढेरा को भारत का पहला 24x7 सौर ऊर्जा संचालित गांव भी घोषित किया। श्री मोदी ने गुजरात के मोढेरा में मोधेश्वरी माता मंदिर में भी दर्शन और पूजा की।""" ]

en_loader = DataLoader(english_texts, batch_size=1, shuffle=True)
hi_loader = DataLoader(hindi_texts, batch_size=1, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scaler = torch.amp.GradScaler("cuda")   # ✅ updated syntax

# Gradient accumulation
accumulation_steps = 8

# -----------------------------
# Helper: Generate synthetic translations (inference only)
# -----------------------------
def generate_translation(batch, src_lang, tgt_lang):
    prompts = [f"Translate {src_lang} to {tgt_lang}: {s}" for s in batch]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)
    return [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]

# -----------------------------
# Helper: Build causal LM loss with masking
# -----------------------------
def build_loss(prompts, targets):
    # Concatenate prompt + target
    inputs = [p + " " + t for p, t in zip(prompts, targets)]
    enc = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    labels = enc["input_ids"].clone()
    for i, p in enumerate(prompts):
        # Tokenize prompt only once per batch
        prompt_ids = tokenizer(p, return_tensors="pt", max_length=512, truncation=True).input_ids.squeeze()
        # Only mask prompt tokens if they fit
        cutoff = min(len(prompt_ids), labels.size(1))
        labels[i, :cutoff] = -100

    enc["labels"] = labels
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):  # ✅ updated syntax
        loss = model(**enc).loss
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
    return loss

# -----------------------------
# Training loop
# -----------------------------
num_epochs = 3

for epoch in range(num_epochs):
    for step, (en_batch, hi_batch) in enumerate(zip(en_loader, hi_loader)):
        # Step A: English mono → HI→EN
        synth_hi = generate_translation(en_batch, "English", "Hindi")
        prompts = [f"Translate the following Hindi text to English:\n\n{text}\n\nTranslation:" for text in synth_hi]
        loss_hi2en = build_loss(prompts, en_batch) / accumulation_steps
        scaler.scale(loss_hi2en).backward()

        # Step B: Hindi mono → EN→HI
        synth_en = generate_translation(hi_batch, "Hindi", "English")
        prompts = [f"Translate the following English text to Hindi:\n\n{text}\n\nTranslation:" for text in synth_en]
        loss_en2hi = build_loss(prompts, hi_batch) / accumulation_steps
        scaler.scale(loss_en2hi).backward()

        # Optimizer step
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    print(f"Epoch {epoch}: Loss_HI2EN={loss_hi2en.item():.4f}, Loss_EN2HI={loss_en2hi.item():.4f}")
