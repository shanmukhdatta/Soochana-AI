"""Faithful code-cell export of event-classification-synthetic-data-generation(1).ipynb.
Execution order follows original notebook. Notebook/Kaggle inputs are required.
This file is a SOURCE EXPORT, not a verified local standalone retraining command.
Do not claim this script was executed during packaging.
"""



# ===== NOTEBOOK CELL 1 =====

# ORIGINAL NOTEBOOK SETUP (install via requirements.txt): %pip -q install -U "groq>=0.33.0" pyarrow scikit-learn



# ===== NOTEBOOK CELL 2 =====

# Imports
import os
import re
import json
import time
import math
import random
import hashlib
import traceback
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from groq import Groq



# ===== NOTEBOOK CELL 3 =====

# ============================================================
# CONFIGURATION — edit only this cell for major runtime settings
# ============================================================

RANDOM_SEED = 42
rng = random.Random(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

TARGETS = {
    "Credit Event": 1500,
    "Merger / Acquisition": 800,
    "Product Launch": 1000,
}

LABEL_TO_ID = {
    "Geopolitical": 0,
    "Macroeconomic": 1,
    "Credit Event": 2,
    "Merger / Acquisition": 3,
    "Product Launch": 4,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
SYNTHETIC_LABELS = {"Credit Event", "Merger / Acquisition", "Product Launch"}

GENERATOR_MODEL = "openai/gpt-oss-120b"
VERIFIER_MODEL = "openai/gpt-oss-20b"

GENERATION_BATCH_SIZE = 15
VERIFICATION_BATCH_SIZE = 12

GENERATOR_REASONING_EFFORT = "medium"
VERIFIER_REASONING_EFFORT = "medium"

GENERATOR_MAX_COMPLETION_TOKENS = 8192
VERIFIER_MAX_COMPLETION_TOKENS = 4096

VERIFIER_CONFIDENCE_THRESHOLD = 0.90
SIMILARITY_THRESHOLD = 0.92

MAX_RETRIES = 5
REQUEST_DELAY_SECONDS = 2.2
BACKOFF_BASE_SECONDS = 3.0

# Safety against infinite loops or extremely poor acceptance.
MAX_BATCHES_PER_CLASS_PER_RUN = 300
MAX_TOTAL_BATCHES_PER_RUN = 700
MAX_CONSECUTIVE_EMPTY_BATCHES = 12

# Real train references are OFF by default to reduce copying risk and token usage.
USE_REAL_TRAIN_REFERENCES = False
REAL_TRAIN_REFERENCES_PER_CALL = 3

# Set True for a quick pipeline check before a full run.
PILOT_MODE = False
PILOT_ACCEPTED_PER_CLASS = 30

# If True, generation runs. Set False to inspect discovered inputs/config only.
RUN_GENERATION = True

WORKDIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
INPUT_ROOT = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path.cwd()

PROMPT_VERSION = "2.0"



# ===== NOTEBOOK CELL 4 =====

# ============================================================
# APPROVED SELF-CONTAINED PROMPT LIBRARY
# ============================================================

GLOBAL_SYSTEM_PROMPT = 'You are a senior financial-markets analyst and synthetic-data author creating TRAINING examples for a five-class financial event classifier:\n\n0 = Geopolitical\n1 = Macroeconomic\n2 = Credit Event\n3 = Merger / Acquisition\n4 = Product Launch\n\nYou will be asked to generate examples for exactly ONE target class per request. Follow the supplied class-specific prompt and batch plan exactly.\n\nMISSION\nCreate original, realistic, financially coherent, unambiguous text whose PRIMARY event clearly belongs to the requested target class. Dataset quality and diversity matter more than raw count.\n\nNON-NEGOTIABLE RULES\n1. One class per request. Every example in the response must belong to the requested target class.\n2. Each example should have exactly one clearly dominant event. Secondary context is allowed only when it cannot change the primary label.\n3. Follow the supplied batch plan slot-by-slot. Each output example must match its assigned subtype, stage, style, difficulty, industry, geography, entity pattern, opening pattern, and amount pattern.\n4. Never leak the class label inside the generated text. Do not write strings such as "Credit Event:", "M&A:", "Product Launch:", "class 2", "class 3", "class 4", "event type", or any annotation.\n5. Do not mention publishers or claim text came from Reuters, Bloomberg, CNBC, or another real publisher.\n6. Prefer fictional but plausible company and institution names. Do not invent factual claims about real listed companies, real executives, or specific real regulatory rulings.\n7. Avoid repeatedly starting with "Company X announced..." or equivalent. Vary syntax, voice, sentence structure, and openings.\n8. Any numbers, currencies, ratings, dates, percentages, prices, coupon amounts, ownership stakes, or deal values must make financial sense.\n9. Do not make Geopolitical or Macroeconomic events the primary event in examples for classes 2, 3, or 4.\n10. Never merely paraphrase previous examples. Do not create superficial variants by changing only company names, countries, dates, or numbers.\n11. Avoid near-duplicate examples inside the same batch.\n12. Output only valid JSON matching the supplied structured-output schema. No Markdown and no commentary.\n\nDIFFICULTY DEFINITIONS\n- easy: the event is stated directly with explicit terminology.\n- medium: natural financial wording with less obvious surface cues, but the event is still clear.\n- difficult: the event is expressed indirectly through consequences, procedure, filings, or market language, but a knowledgeable financial reader can still classify it unambiguously.\n\nSTYLE PRINCIPLES\nWrite original financial language. Do not copy real articles. A "wire-like" style means concise factual writing only; it must not imitate a named publisher.\n\nBefore answering, silently verify every example:\n(a) Is the intended target event genuinely present?\n(b) Is another of the five classes more appropriate?\n(c) Is the text ambiguous?\n(d) Does it contain label leakage?\n(e) Is it a superficial template variation of another example?\nRewrite any example that fails.'
CREDIT_EVENT_GENERATION_PROMPT = 'BATCH_ID: <<BATCH_ID>>\nTARGET_CLASS: Credit Event\nTARGET_LABEL_ID: 2\nREQUESTED_COUNT: <<REQUESTED_COUNT>>\n\nTASK\nGenerate exactly REQUESTED_COUNT original examples whose primary event is a CORPORATE OR INSTITUTIONAL CREDIT EVENT.\n\nSTRICT DEFINITION\nA Credit Event is genuine deterioration, distress, default, insolvency, or a material adverse change in the creditworthiness of a company, bank, insurer, corporate issuer, institutional borrower, or its debt obligation.\n\nVALID SUBTYPES\n- missed_interest_payment\n- missed_principal_payment\n- bond_default\n- loan_default\n- bankruptcy_filing\n- insolvency\n- receivership_administration\n- distressed_debt_restructuring\n- covenant_breach\n- debt_acceleration\n- liquidity_crisis_threatening_repayment\n- credit_rating_downgrade\n- negative_watch_or_outlook_credit\n- distress_restructuring_negotiations\n- payment_moratorium\n- debt_standstill\n\nVALID EVENT STAGES\n- warning_signs\n- grace_period\n- missed\n- acknowledged\n- creditor_negotiation\n- forbearance\n- filed\n- court_process\n- restructuring_agreed\n- downgraded\n- watch_placed\n- resolved_after_distress\n\nWHAT A VALID POSITIVE MUST SHOW\nThe text itself must contain enough evidence to justify the credit label, such as:\n- an interest or principal payment that was missed or not remitted;\n- a grace period expiring without payment;\n- a bankruptcy, insolvency, receivership, or administration filing;\n- lenders accelerating debt after a contractual trigger;\n- a material covenant breach;\n- creditors negotiating because repayment capacity has deteriorated;\n- a distressed restructuring, payment standstill, or moratorium;\n- a liquidity shortfall that explicitly threatens repayment;\n- a credit-rating downgrade or negative watch/outlook tied to weakened debt-service capacity or creditworthiness.\n\nVALID WORDING VARIETY\nDo not make every example use the word "default". Vary natural wording such as:\nfailed to remit, skipped a coupon, grace period expired, sought creditor protection, entered administration, lenders demanded immediate repayment, requested a waiver after a ratio breach, was cut to speculative grade, outlook moved negative on weakening debt-service capacity, agreed a standstill with bondholders, suspended creditor payments, or began restructuring after failing to refinance.\n\nDO NOT GENERATE THESE AS CREDIT EVENT POSITIVES\n- ordinary bond issuance or loan raising;\n- bond-price or bond-yield movement without credit impairment;\n- a simple increase in total debt;\n- routine refinancing;\n- ordinary capital raising;\n- higher interest expense by itself;\n- equity analyst Buy/Sell/Hold recommendations;\n- ordinary stock-price weakness;\n- sovereign default or sovereign downgrade;\n- a central-bank decision, inflation release, GDP report, recession forecast, or other macro event.\n\nBOUNDARY RULES\n1. If debt is mentioned, show actual impairment, distress, breach, missed payment, restructuring, or creditworthiness deterioration. Mere debt existence is not enough.\n2. If a rating event is used, make clear it is a CREDIT rating action on a corporate/institutional issuer or debt instrument, and state the direction/reason.\n3. Do not make a merger, acquisition, product release, geopolitical action, or macroeconomic event the competing primary event.\n4. "Company raises $500 million through a bond issue" is NOT a Credit Event.\n5. "Company misses a $500 million bond payment after its grace period expires" IS a Credit Event.\n6. "Bond yields rose" is NOT enough.\n7. "Lenders accelerated the loan after a leverage-covenant breach" IS a Credit Event.\n\nBATCH PLAN\nGenerate one example for each slot below, in order:\n<<BATCH_PLAN_JSON>>\n\nEach slot can specify:\nsubtype, event_stage, style, difficulty, industry, geography, scenario_family, entity_pattern, opening_pattern, amount_pattern.\n\nSTYLE DEFINITIONS\n- headline: 8-25 words, compact financial-news headline/alert.\n- news_sentence_paragraph: one sentence of about 15-45 words or a short paragraph of about 35-100 words.\n- wire_concise: concise fact-first report of about 20-60 words.\n- social_post: market/social-finance style, roughly 12-60 words, still financially clear and non-sensational.\n- filing_style: formal corporate/regulatory disclosure wording, roughly 30-100 words.\n- indirect_contextual: roughly 25-100 words, event conveyed through consequences/procedure but still unambiguous.\n\nDIFFICULTY TARGETS\neasy = explicit event terminology.\nmedium = clear event with less obvious lexical cues.\ndifficult = indirect but still unambiguous to a financially knowledgeable reader.\n\nDIVERSITY REQUIREMENTS\n- Do not reuse the same sentence skeleton.\n- Do not merely swap company names, amounts, countries, or dates.\n- Vary industries, borrower types, regions, currencies, debt instruments, stages, and narrative perspective.\n- At most one example in the batch may begin with a company name.\n- Do not repeat the same fictional company name, exact amount, or percentage in the batch.\n- Avoid overusing these fictional-name stems: Apex, Nova, Nexus, Quantum, Zenith, Vertex, Orion, Titan, Pinnacle, Summit, Global, Sterling, Atlas, Phoenix, Horizon, Vanguard, Stellar, Fusion, Alpha.\n- Make scenario_family a short snake_case description of the event skeleton. Every example in the batch must have a different scenario_family.\n- Do not use "default" as the defining keyword in most examples.\n\nPREVIOUSLY USED SCENARIO FINGERPRINTS\nDo not repeat or closely resemble these:\n<<PREVIOUS_FINGERPRINTS>>\n\nOPTIONAL REAL TRAIN REFERENCES\nUse only for general domain/register orientation. Never copy, closely paraphrase, reuse names/numbers, or create near-duplicates:\n<<REAL_TRAIN_REFERENCES>>\n\nOUTPUT REQUIREMENTS\nReturn valid JSON only.\nEvery example must contain exactly these semantic fields:\ntext, label, label_id, subtype, style, difficulty, event_stage, industry, geography, scenario_family.\nFor every example:\nlabel must be exactly "Credit Event"\nlabel_id must be exactly 2.'
MERGER_ACQUISITION_GENERATION_PROMPT = 'BATCH_ID: <<BATCH_ID>>\nTARGET_CLASS: Merger / Acquisition\nTARGET_LABEL_ID: 3\nREQUESTED_COUNT: <<REQUESTED_COUNT>>\n\nTASK\nGenerate exactly REQUESTED_COUNT original examples whose primary event is a MERGER, ACQUISITION, TAKEOVER, BUSINESS DIVESTITURE, OR OTHER CONTROL/OWNERSHIP TRANSACTION.\n\nSTRICT DEFINITION\nA Merger / Acquisition event occurs when control or ownership of a business, business unit, or controlling asset changes or is explicitly being pursued, proposed, agreed, reviewed, approved, completed, rejected, withdrawn, or terminated.\n\nVALID SUBTYPES\n- acquisition\n- merger\n- takeover\n- takeover_bid\n- tender_offer\n- business_unit_acquisition\n- controlling_asset_acquisition\n- divestiture_business_sale\n- merger_agreement\n- acquisition_agreement\n- regulatory_approval_of_deal\n- shareholder_approval_of_deal\n- completed_acquisition\n- rejected_takeover_offer\n- terminated_or_cancelled_deal\n- rumored_or_considered_deal\n\nVALID EVENT STAGES\n- rumored\n- considering\n- proposed\n- announced\n- signed\n- shareholder_approved\n- pending_regulatory_review\n- regulatory_approved\n- completed\n- rejected\n- withdrawn\n- terminated\n\nWHAT A VALID POSITIVE MUST SHOW\nThe text must clearly describe an ownership/control transaction or its lifecycle. It should identify, explicitly or unambiguously, a buyer/bidder/merging party/seller and a target business, business unit, or controlling stake/asset.\n\nRumored or considered cases qualify only when M&A intent is explicit, for example:\nexploring a sale, weighing a takeover bid, approached about a buyout, considering a merger, seeking buyers for a division.\n\nVALID WORDING VARIETY\nDo not make every example use "acquire". Vary natural wording such as:\nbuy out, take private, combine operations, absorb, merge, purchase a controlling stake, carve out and sell, divest a unit, tender for shares, make an unsolicited offer, all-share combination, scheme of arrangement, bid lapsed, offer rejected, transaction cleared, closing conditions satisfied, deal collapsed, or sale completed.\n\nDO NOT GENERATE THESE AS M&A POSITIVES\n- ordinary minority/passive investment without control;\n- venture-capital or growth-funding round;\n- strategic partnership;\n- joint marketing agreement;\n- supplier/distribution agreement;\n- ordinary equipment/property purchase;\n- ordinary capital raising;\n- share buyback;\n- customer contract;\n- licensing arrangement without transfer of business control.\n\nBOUNDARY RULES\n1. A stake purchase qualifies only when control, a majority/controlling stake, takeover intent, or equivalent change in ownership/control is clear.\n2. "Company A invests $2 billion in Company B" is NOT automatically M&A.\n3. "Company A agrees to acquire Company B for $2 billion" IS M&A.\n4. If debt financing is mentioned, keep it secondary to the ownership transaction.\n5. Do not make a Credit Event, Product Launch, Geopolitical event, or Macroeconomic event the competing primary event.\n6. Partnership language alone must not be converted into M&A.\n7. Ordinary purchase of machinery, property, inventory, or raw materials is not M&A unless it is clearly an acquisition of a business/control asset.\n\nBATCH PLAN\nGenerate one example for each slot below, in order:\n<<BATCH_PLAN_JSON>>\n\nEach slot can specify:\nsubtype, event_stage, style, difficulty, industry, geography, scenario_family, entity_pattern, opening_pattern, amount_pattern.\n\nSTYLE DEFINITIONS\n- headline: 8-25 words, compact financial-news headline/alert.\n- news_sentence_paragraph: one sentence of about 15-45 words or a short paragraph of about 35-100 words.\n- wire_concise: concise fact-first report of about 20-60 words.\n- social_post: market/social-finance style, roughly 12-60 words, still financially clear.\n- filing_style: formal corporate/regulatory disclosure wording, roughly 30-100 words.\n- indirect_contextual: roughly 25-100 words, transaction conveyed indirectly through procedure/consequences but still unambiguous.\n\nDIFFICULTY TARGETS\neasy = explicit merger/acquisition/takeover terminology.\nmedium = natural financial wording with less obvious lexical cues.\ndifficult = transaction inferred from clear control/ownership procedure or consequences, but still unambiguous.\n\nDIVERSITY REQUIREMENTS\n- Do not reuse the same transaction skeleton.\n- Do not merely substitute company names, deal values, ownership percentages, countries, or dates.\n- Vary buyer/target/seller/regulator/shareholder perspectives.\n- Include small, mid-sized, and large transactions across many sectors and regions.\n- Include announced, pending, approved, completed, rejected, withdrawn, and terminated transactions.\n- At most one example in the batch may begin with a company name.\n- Never repeat the same company name, exact amount, or ownership percentage in the batch.\n- Avoid overused fictional-name stems: Apex, Nova, Nexus, Quantum, Zenith, Vertex, Orion, Titan, Pinnacle, Summit, Global, Sterling, Atlas, Phoenix, Horizon, Vanguard, Stellar, Fusion, Alpha.\n- Every example must have a unique snake_case scenario_family.\n- Do not use "acquire" or "acquisition" as the defining word in most examples.\n\nPREVIOUSLY USED SCENARIO FINGERPRINTS\nDo not repeat or closely resemble these:\n<<PREVIOUS_FINGERPRINTS>>\n\nOPTIONAL REAL TRAIN REFERENCES\nUse only for general domain/register orientation. Never copy, closely paraphrase, reuse names/numbers, or create near-duplicates:\n<<REAL_TRAIN_REFERENCES>>\n\nOUTPUT REQUIREMENTS\nReturn valid JSON only.\nEvery example must contain exactly these semantic fields:\ntext, label, label_id, subtype, style, difficulty, event_stage, industry, geography, scenario_family.\nFor every example:\nlabel must be exactly "Merger / Acquisition"\nlabel_id must be exactly 3.'
PRODUCT_LAUNCH_GENERATION_PROMPT = 'BATCH_ID: <<BATCH_ID>>\nTARGET_CLASS: Product Launch\nTARGET_LABEL_ID: 4\nREQUESTED_COUNT: <<REQUESTED_COUNT>>\n\nTASK\nGenerate exactly REQUESTED_COUNT original examples whose primary event is the INTRODUCTION, UNVEILING, RELEASE, COMMERCIAL AVAILABILITY, ROLLOUT, DELAY, OR CANCELLATION OF A GENUINELY NEW OR NEWLY COMMERCIALIZED PRODUCT, SERVICE, OR PLATFORM.\n\nSTRICT DEFINITION\nA Product Launch event requires a specific new offering, or a meaningful new-market introduction of that offering, to be announced, unveiled, opened for preorder, released, made commercially available, rolled out, delayed, or cancelled.\n\nVALID SUBTYPES\n- physical_product_launch\n- software_product_launch\n- ai_product_service_launch\n- financial_product_service_launch\n- digital_platform_launch\n- enterprise_service_launch\n- commercial_release\n- public_availability\n- product_unveiling\n- rollout\n- geographic_market_launch\n- new_service_launch\n- delayed_launch\n- cancelled_launch\n- beta_to_commercial_launch\n- next_generation_product_introduction\n\nVALID EVENT STAGES\n- preview\n- unveiling\n- announcement\n- pre_order\n- initial_release\n- commercial_availability\n- market_rollout\n- regional_expansion\n- delay\n- cancellation\n\nWHAT A VALID POSITIVE MUST SHOW\nThe text must clearly describe a specific new offering and at least one meaningful launch signal:\nwhat is being introduced, when it becomes available, where it becomes available, who can access/buy it, a preorder/release step, movement from beta to commercial availability, or a clearly identified planned launch being delayed/cancelled.\n\nVALID WORDING VARIETY\nDo not make every example use "launch". Vary natural wording such as:\nunveils, debuts, introduces, begins shipping, opens preorders, goes on sale, makes generally available, rolls out, takes the service live, exits beta, starts commercial deliveries, extends availability to a new market, pushes back release, or shelves the planned rollout.\n\nDO NOT GENERATE THESE AS PRODUCT-LAUNCH POSITIVES\n- higher sales of an existing product;\n- product price changes;\n- recalls;\n- awards;\n- earnings commentary that merely mentions an existing product;\n- partnerships with no new offering becoming available;\n- routine maintenance, patching, or minor updates;\n- vague R&D discussion with no concrete release/availability event;\n- analyst commentary on an existing product;\n- financial-instrument issuance such as a bond offering or share offering.\n\nBOUNDARY RULES\n1. "Strong sales of the company\'s new phone" is NOT by itself a launch event.\n2. "The company unveiled the phone and will begin sales Friday" IS a Product Launch.\n3. "The company launches a $1 billion bond offering" is NOT Product Launch.\n4. "The bank rolls out a new digital-payments platform to retail customers" IS Product Launch.\n5. Routine bug fixes, maintenance releases, and ordinary version patches are NOT Product Launch unless the text clearly describes a materially new commercial offering.\n6. A partnership is not Product Launch unless the primary event is a concrete new product/service becoming available.\n7. Do not make an acquisition, Credit Event, Geopolitical event, or Macroeconomic event the competing primary event.\n\nBATCH PLAN\nGenerate one example for each slot below, in order:\n<<BATCH_PLAN_JSON>>\n\nEach slot can specify:\nsubtype, event_stage, style, difficulty, industry, geography, scenario_family, entity_pattern, opening_pattern, amount_pattern.\n\nSTYLE DEFINITIONS\n- headline: 8-25 words, compact financial-news headline/alert.\n- news_sentence_paragraph: one sentence of about 15-45 words or a short paragraph of about 35-100 words.\n- wire_concise: concise fact-first report of about 20-60 words.\n- social_post: market/social-finance style, roughly 12-60 words, still financially clear.\n- filing_style: formal corporate disclosure wording, roughly 30-100 words.\n- indirect_contextual: roughly 25-100 words, release/availability event expressed through consequences/procedure but still unambiguous.\n\nDIFFICULTY TARGETS\neasy = explicit launch/release/unveiling terminology.\nmedium = natural product/commercial wording with fewer obvious lexical cues.\ndifficult = availability/introduction is expressed indirectly but a knowledgeable reader can still classify it confidently.\n\nDIVERSITY REQUIREMENTS\n- Do not reuse the same product-release skeleton.\n- Do not merely replace product names, prices, dates, geographies, or company names.\n- Cover technology, banking/fintech, automotive, energy, healthcare, industrials, telecom, consumer goods, retail, logistics, software, AI, and other sectors.\n- Include physical goods, software, services, platforms, financial products/services, regional rollouts, beta-to-commercial moves, delays, and cancellations.\n- At most one example per batch may begin with a company name.\n- Never repeat the same fictional company name, exact price, or release date in the batch.\n- Avoid overused fictional-name stems: Apex, Nova, Nexus, Quantum, Zenith, Vertex, Orion, Titan, Pinnacle, Summit, Global, Sterling, Atlas, Phoenix, Horizon, Vanguard, Stellar, Fusion, Alpha.\n- Every example must have a unique snake_case scenario_family.\n- Do not use "launch" as the defining keyword in most examples.\n\nPREVIOUSLY USED SCENARIO FINGERPRINTS\nDo not repeat or closely resemble these:\n<<PREVIOUS_FINGERPRINTS>>\n\nOPTIONAL REAL TRAIN REFERENCES\nUse only for general domain/register orientation. Never copy, closely paraphrase, reuse names/numbers, or create near-duplicates:\n<<REAL_TRAIN_REFERENCES>>\n\nOUTPUT REQUIREMENTS\nReturn valid JSON only.\nEvery example must contain exactly these semantic fields:\ntext, label, label_id, subtype, style, difficulty, event_stage, industry, geography, scenario_family.\nFor every example:\nlabel must be exactly "Product Launch"\nlabel_id must be exactly 4.'
VERIFIER_SYSTEM_PROMPT = 'You are a strict, conservative quality auditor for a five-class financial-event classification dataset.\n\nFULL CLASS TAXONOMY\n\n0 = Geopolitical\nPrimary event involves cross-border political or state action materially affecting financial/economic conditions, such as sanctions, export controls, armed interstate conflict, trade restrictions/tariffs, diplomatic escalation, or international political instability.\nDo NOT use Geopolitical for ordinary domestic corporate transactions, ordinary macroeconomic releases, company defaults, M&A, or product introductions.\n\n1 = Macroeconomic\nPrimary event involves economy-wide conditions or policy such as central-bank rate decisions, inflation, GDP/growth, employment, recession/outlook, fiscal-policy changes, broad FX/macroeconomic developments, or other economy-wide indicators.\nDo NOT use Macroeconomic for a single company\'s credit distress, ownership transaction, or product release unless the company event is clearly secondary to the economy-wide event.\n\n2 = Credit Event\nPrimary event is corporate/institutional credit deterioration: missed interest/principal payment, default, bankruptcy/insolvency/receivership, covenant breach, debt acceleration, liquidity crisis threatening repayment, distressed restructuring, creditor standstill/moratorium, or corporate credit-rating downgrade/negative watch tied to creditworthiness.\nNOT Credit Event: ordinary bond issuance, yield/price changes, normal refinancing, debt growth alone, equity-analyst ratings.\n\n3 = Merger / Acquisition\nPrimary event changes or explicitly pursues ownership/control of a business, business unit, or controlling stake/asset: acquisition, merger, takeover/bid, tender offer, divestiture/business sale, agreement, approval, completion, rejection, withdrawal, or termination.\nNOT M&A: passive minority investment without control, venture funding, partnership, supplier agreement, capital raising, ordinary equipment purchase.\n\n4 = Product Launch\nPrimary event is the introduction, unveiling, release, commercial availability, meaningful rollout, delay, or cancellation of a genuinely new/newly commercialized product, service, or platform.\nNOT Product Launch: existing-product sales, price changes, recalls, awards, routine maintenance, vague R&D, ordinary earnings mentions, bond/share offerings.\n\nNO-EVENT SENTINEL\nIf none of the five classes clearly fits, use predicted_label = "None" and predicted_label_id = -1.\n"None" is a verifier-only rejection sentinel and is NEVER a training label.\n\nINDEPENDENT VERIFICATION PROCEDURE\nFor each candidate:\n1. Read the text and decide the best class from its content BEFORE considering the intended label.\n2. Compare your independent classification with the intended label.\n3. Check whether the intended event is actually present.\n4. Check whether another class is more appropriate.\n5. Check ambiguity: could a knowledgeable financial reader reasonably disagree about the primary class?\n6. Check whether the language is financially realistic.\n7. Check for artificial label leakage or meta annotations.\n8. Check whether it is too generic to train on.\n9. Check whether it appears to be a trivial/template variation within the supplied verification batch.\n10. Check whether multiple competing events make the primary label unclear.\n\nDECISION RULES\n- Be conservative. When genuinely uncertain, reject.\n- reason_code must be PASS only when accept is true.\n- If accept is false, choose the single most important failure reason.\n- Confidence is the probability from 0 to 1 that the text belongs to the INTENDED class under this taxonomy and is unambiguous.\n- Use confidence >= 0.90 only for clear cases.\n- Do not automatically agree with the supplied label.\n- Output JSON only and follow the supplied schema exactly.'
VERIFIER_BATCH_PROMPT = 'Audit every candidate below.\n\nFor each candidate:\n1. classify the text independently using the five-class taxonomy;\n2. compare your independent prediction with intended_label/intended_label_id;\n3. decide whether it should be accepted.\n\nReturn exactly one result for every candidate_id, preserving candidate order.\n\nA candidate may be accepted ONLY when ALL are true:\n- the intended event is genuinely present;\n- predicted_label equals intended_label;\n- predicted_label_id equals intended_label_id;\n- confidence >= 0.90;\n- ambiguity is false;\n- language is realistic and sufficiently specific;\n- there is no label leakage;\n- there is no competing primary event;\n- reason_code is PASS.\n\nAllowed reason_code values EXACTLY:\nPASS\nWRONG_CLASS\nAMBIGUOUS\nNO_EVENT\nUNREALISTIC\nLABEL_LEAKAGE\nTOO_GENERIC\nLIKELY_TEMPLATE_DUPLICATE\nMULTIPLE_COMPETING_EVENTS\n\nDo not automatically relabel failed candidates for training. A failed candidate remains rejected.\n\nCANDIDATES_JSON:\n<<CANDIDATES_JSON>>\n\nReturn JSON only, matching the verifier schema.'

APPROVED_CONFIG = {'version': '2.0', 'full_label_mapping': {'Geopolitical': 0, 'Macroeconomic': 1, 'Credit Event': 2, 'Merger / Acquisition': 3, 'Product Launch': 4}, 'synthetic_target_labels': {'Credit Event': 2, 'Merger / Acquisition': 3, 'Product Launch': 4}, 'accepted_targets': {'Credit Event': 1500, 'Merger / Acquisition': 800, 'Product Launch': 1000}, 'generator': {'model': 'openai/gpt-oss-120b', 'reasoning_effort': 'medium', 'temperature_default': 1.0, 'temperature_rotation': [0.8, 1.0, 1.1], 'top_p': 0.95, 'generation_batch_size': 15, 'max_batch_size': 20, 'max_completion_tokens': 8192, 'strict_json_schema': True, 'max_previous_fingerprints': 60, 'real_train_references_per_call_max': 4}, 'verifier': {'model': 'openai/gpt-oss-20b', 'reasoning_effort': 'medium', 'temperature': 0.1, 'top_p': 1.0, 'verification_batch_size': 12, 'max_completion_tokens': 4096, 'verifier_confidence_threshold': 0.9}, 'acceptance': {'accept_must_be_true': True, 'predicted_label_must_match': True, 'predicted_label_id_must_match': True, 'minimum_confidence': 0.9, 'ambiguity_must_be_false': True, 'reason_code_must_be': 'PASS'}, 'style_mix': {'headline': 0.3, 'news_sentence_paragraph': 0.25, 'wire_concise': 0.15, 'social_post': 0.15, 'filing_style': 0.1, 'indirect_contextual': 0.05}, 'difficulty_mix': {'easy': 0.35, 'medium': 0.4, 'difficult': 0.25}, 'dedupe': {'near_duplicate_similarity_threshold': 0.92, 'compare_against': ['real_train', 'accepted_synthetic', 'current_batch']}, 'data_hygiene': {'synthetic_in_validation': False, 'synthetic_in_test': False, 'real_validation_as_generation_reference': False, 'real_test_as_generation_reference': False}}
CREDIT_TAXONOMY = {'label': 'Credit Event', 'label_id': 2, 'subtype_weights': {'missed_interest_payment': 8, 'missed_principal_payment': 6, 'bond_default': 8, 'loan_default': 7, 'bankruptcy_filing': 10, 'insolvency': 6, 'receivership_administration': 5, 'distressed_debt_restructuring': 8, 'covenant_breach': 8, 'debt_acceleration': 4, 'liquidity_crisis_threatening_repayment': 7, 'credit_rating_downgrade': 10, 'negative_watch_or_outlook_credit': 5, 'distress_restructuring_negotiations': 4, 'payment_moratorium': 2, 'debt_standstill': 2}, 'stage_weights': {'warning_signs': 7, 'grace_period': 7, 'missed': 14, 'acknowledged': 7, 'creditor_negotiation': 11, 'forbearance': 7, 'filed': 12, 'court_process': 8, 'restructuring_agreed': 8, 'downgraded': 10, 'watch_placed': 4, 'resolved_after_distress': 5}, 'styles': ['headline', 'news_sentence_paragraph', 'wire_concise', 'social_post', 'filing_style', 'indirect_contextual'], 'difficulty_levels': ['easy', 'medium', 'difficult'], 'scenario_families': ['coupon_miss_after_grace_period', 'principal_payment_not_remitted', 'term_loan_payment_default', 'bankruptcy_protection_filing', 'court_appointed_administration', 'liquidity_shortfall_threatens_debt_service', 'leverage_covenant_breach_and_waiver', 'debt_accelerated_after_breach', 'distressed_exchange_with_bondholders', 'creditor_standstill_agreement', 'payment_moratorium_during_restructuring', 'rating_cut_on_weak_debt_service', 'negative_watch_on_refinancing_risk', 'creditor_negotiation_after_failed_refinancing', 'receivership_after_unpaid_secured_loan', 'distress_resolved_after_restructuring']}
MA_TAXONOMY = {'label': 'Merger / Acquisition', 'label_id': 3, 'subtype_weights': {'acquisition': 14, 'merger': 9, 'takeover': 6, 'takeover_bid': 6, 'tender_offer': 5, 'business_unit_acquisition': 8, 'controlling_asset_acquisition': 5, 'divestiture_business_sale': 9, 'merger_agreement': 5, 'acquisition_agreement': 5, 'regulatory_approval_of_deal': 6, 'shareholder_approval_of_deal': 4, 'completed_acquisition': 8, 'rejected_takeover_offer': 4, 'terminated_or_cancelled_deal': 4, 'rumored_or_considered_deal': 2}, 'stage_weights': {'rumored': 6, 'considering': 6, 'proposed': 10, 'announced': 18, 'signed': 12, 'shareholder_approved': 6, 'pending_regulatory_review': 10, 'regulatory_approved': 7, 'completed': 13, 'rejected': 5, 'withdrawn': 3, 'terminated': 4}, 'styles': ['headline', 'news_sentence_paragraph', 'wire_concise', 'social_post', 'filing_style', 'indirect_contextual'], 'difficulty_levels': ['easy', 'medium', 'difficult'], 'scenario_families': ['cash_takeover_offer', 'all_share_merger', 'majority_stake_purchase', 'remaining_shares_buyout', 'private_equity_take_private', 'business_unit_carveout_sale', 'subsidiary_divestiture', 'controlling_asset_transaction', 'tender_offer_for_control', 'hostile_bid_rejected', 'board_recommends_offer', 'shareholders_approve_combination', 'competition_authority_clears_deal', 'deal_pending_regulatory_review', 'closing_conditions_satisfied', 'transaction_completed', 'bid_withdrawn', 'agreement_terminated', 'company_explores_sale', 'buyer_weighs_takeover_bid']}
PRODUCT_TAXONOMY = {'label': 'Product Launch', 'label_id': 4, 'subtype_weights': {'physical_product_launch': 9, 'software_product_launch': 8, 'ai_product_service_launch': 10, 'financial_product_service_launch': 8, 'digital_platform_launch': 8, 'enterprise_service_launch': 6, 'commercial_release': 6, 'public_availability': 6, 'product_unveiling': 7, 'rollout': 6, 'geographic_market_launch': 8, 'new_service_launch': 5, 'delayed_launch': 4, 'cancelled_launch': 3, 'beta_to_commercial_launch': 3, 'next_generation_product_introduction': 3}, 'stage_weights': {'preview': 6, 'unveiling': 12, 'announcement': 14, 'pre_order': 8, 'initial_release': 14, 'commercial_availability': 16, 'market_rollout': 12, 'regional_expansion': 10, 'delay': 4, 'cancellation': 4}, 'styles': ['headline', 'news_sentence_paragraph', 'wire_concise', 'social_post', 'filing_style', 'indirect_contextual'], 'difficulty_levels': ['easy', 'medium', 'difficult'], 'scenario_families': ['hardware_unveiling_with_sale_date', 'software_general_availability', 'ai_service_commercial_release', 'digital_platform_goes_live', 'new_bank_account_or_card_rollout', 'payments_service_customer_release', 'enterprise_service_release', 'preorders_open_for_new_product', 'first_commercial_shipments', 'beta_exits_to_paid_release', 'regional_market_expansion', 'new_vehicle_model_sales_begin', 'industrial_equipment_commercial_release', 'medical_device_commercial_availability', 'consumer_product_debut', 'telecom_service_rollout', 'launch_delayed_before_release', 'planned_rollout_cancelled', 'next_generation_product_introduction']}

CLASS_PROMPTS = {
    "Credit Event": CREDIT_EVENT_GENERATION_PROMPT,
    "Merger / Acquisition": MERGER_ACQUISITION_GENERATION_PROMPT,
    "Product Launch": PRODUCT_LAUNCH_GENERATION_PROMPT,
}

CLASS_TAXONOMIES = {
    "Credit Event": CREDIT_TAXONOMY,
    "Merger / Acquisition": MA_TAXONOMY,
    "Product Launch": PRODUCT_TAXONOMY,
}



# ===== NOTEBOOK CELL 5 =====

# Structured-output schemas

GENERATOR_STYLES = [
    "headline",
    "news_sentence_paragraph",
    "wire_concise",
    "social_post",
    "filing_style",
    "indirect_contextual",
]
GENERATOR_DIFFICULTIES = ["easy", "medium", "difficult"]

def build_generator_response_format(label, label_id, subtypes, stages):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "synthetic_event_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["batch_id", "examples"],
                "properties": {
                    "batch_id": {"type": "string"},
                    "examples": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "text", "label", "label_id", "subtype", "style",
                                "difficulty", "event_stage", "industry", "geography",
                                "scenario_family",
                            ],
                            "properties": {
                                "text": {"type": "string"},
                                "label": {"type": "string", "enum": [label]},
                                "label_id": {"type": "integer", "enum": [label_id]},
                                "subtype": {"type": "string", "enum": subtypes},
                                "style": {"type": "string", "enum": GENERATOR_STYLES},
                                "difficulty": {"type": "string", "enum": GENERATOR_DIFFICULTIES},
                                "event_stage": {"type": "string", "enum": stages},
                                "industry": {"type": "string"},
                                "geography": {"type": "string"},
                                "scenario_family": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    }

VERIFIER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "verification_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["results"],
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "candidate_id", "predicted_label", "predicted_label_id",
                            "accept", "confidence", "ambiguity", "reason_code",
                        ],
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "predicted_label": {
                                "type": "string",
                                "enum": [
                                    "Geopolitical", "Macroeconomic", "Credit Event",
                                    "Merger / Acquisition", "Product Launch", "None"
                                ],
                            },
                            "predicted_label_id": {
                                "type": "integer",
                                "enum": [-1, 0, 1, 2, 3, 4],
                            },
                            "accept": {"type": "boolean"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "ambiguity": {"type": "boolean"},
                            "reason_code": {
                                "type": "string",
                                "enum": [
                                    "PASS", "WRONG_CLASS", "AMBIGUOUS", "NO_EVENT",
                                    "UNREALISTIC", "LABEL_LEAKAGE", "TOO_GENERIC",
                                    "LIKELY_TEMPLATE_DUPLICATE", "MULTIPLE_COMPETING_EVENTS",
                                ],
                            },
                        },
                    },
                },
            },
        },
    },
}



# ===== NOTEBOOK CELL 6 =====

# ============================================================
# Discover previous preprocessing outputs under /kaggle/input
# ============================================================

def find_candidates(root: Path, exact_names):
    found = []
    if not root.exists():
        return found
    for name in exact_names:
        found.extend(root.rglob(name))
    return sorted(set(found), key=lambda p: str(p))

def resolve_preferred_file(stem):
    parquet_name = f"{stem}.parquet"
    csv_name = f"{stem}.csv"

    pq = find_candidates(INPUT_ROOT, [parquet_name])
    csv = find_candidates(INPUT_ROOT, [csv_name])

    print(f"\nCandidates for {stem}:")
    for p in pq + csv:
        print("  ", p)

    if pq:
        chosen = pq[0]
    elif csv:
        chosen = csv[0]
    else:
        chosen = None

    print("Selected:", chosen)
    return chosen

TRAIN_PATH = resolve_preferred_file("event_train")
VAL_PATH = resolve_preferred_file("event_validation")
TEST_PATH = resolve_preferred_file("event_test")

if TRAIN_PATH is None:
    # Also allow already-renamed real/frozen files from a prior pipeline.
    alt = find_candidates(INPUT_ROOT, [
        "event_train_real.parquet", "event_train_real.csv",
        "event_train_real_frozen.parquet", "event_train_real_frozen.csv",
    ])
    if alt:
        TRAIN_PATH = sorted(alt, key=lambda p: (p.suffix != ".parquet", str(p)))[0]
        print("\nUsing alternate real-train file:", TRAIN_PATH)

if TRAIN_PATH is None:
    raise FileNotFoundError(
        "Could not locate event_train.parquet/event_train.csv under /kaggle/input. "
        "Attach the output of the previous preprocessing notebook as a Kaggle Input."
    )

def load_table(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")

train_real = load_table(TRAIN_PATH)
val_real = load_table(VAL_PATH) if VAL_PATH else None
test_real = load_table(TEST_PATH) if TEST_PATH else None

print("\nResolved input paths")
print("TRAIN:", TRAIN_PATH)
print("VAL  :", VAL_PATH)
print("TEST :", TEST_PATH)
print("\nTrain shape:", train_real.shape)
if val_real is not None:
    print("Validation shape:", val_real.shape)
if test_real is not None:
    print("Test shape:", test_real.shape)



# ===== NOTEBOOK CELL 7 =====

# ============================================================
# Validate schema and taxonomy
# ============================================================

REQUIRED_TRAIN_COLUMNS = {"text", "label", "label_id"}
missing_cols = REQUIRED_TRAIN_COLUMNS - set(train_real.columns)
if missing_cols:
    raise ValueError(f"REAL train is missing required columns: {sorted(missing_cols)}")

def validate_real_split(df, name):
    if df is None:
        return
    miss = REQUIRED_TRAIN_COLUMNS - set(df.columns)
    if miss:
        raise ValueError(f"{name} missing columns: {sorted(miss)}")

    if df["text"].isna().any() or (df["text"].astype(str).str.strip() == "").any():
        raise ValueError(f"{name} contains null/empty text.")

    labels = set(df["label"].astype(str).unique())
    bad_labels = labels - set(LABEL_TO_ID)
    if bad_labels:
        raise ValueError(f"{name} contains unexpected labels: {sorted(bad_labels)}")

    pairs = df[["label", "label_id"]].drop_duplicates()
    for row in pairs.itertuples(index=False):
        label = str(row.label)
        label_id = int(row.label_id)
        if LABEL_TO_ID[label] != label_id:
            raise ValueError(
                f"{name} has inconsistent mapping: {label!r} -> {label_id}, "
                f"expected {LABEL_TO_ID[label]}"
            )

validate_real_split(train_real, "train_real")
validate_real_split(val_real, "val_real")
validate_real_split(test_real, "test_real")

print("REAL train class counts:")
display(train_real["label"].value_counts().reindex(LABEL_TO_ID.keys(), fill_value=0).rename("count").to_frame())

if val_real is None or test_real is None:
    print("\nWARNING: Validation/test were not discovered. Synthetic generation can continue, "
          "but keep the previous real validation/test outputs for the later FinBERT notebook.")
else:
    print("\nValidation/test located and will NEVER be used as generation references.")



# ===== NOTEBOOK CELL 8 =====

# ============================================================
# Groq API key + client
# ============================================================

def get_groq_api_key():
    key = None
    try:
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("GROQ_API_KEY")
    except Exception:
        key = None

    if not key:
        key = os.environ.get("GROQ_API_KEY")

    if not key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Add a Kaggle Secret named GROQ_API_KEY "
            "and enable notebook access to it."
        )
    return key

GROQ_API_KEY = get_groq_api_key()
client = Groq(api_key=GROQ_API_KEY)
print("Groq client initialized. API key is not printed.")



# ===== NOTEBOOK CELL 9 =====

# ============================================================
# Batch-plan taxonomies and compatibility maps
# ============================================================

STYLE_WEIGHTS = {
    "headline": 30,
    "news_sentence_paragraph": 25,
    "wire_concise": 15,
    "social_post": 15,
    "filing_style": 10,
    "indirect_contextual": 5,
}
DIFFICULTY_WEIGHTS = {"easy": 35, "medium": 40, "difficult": 25}

INDUSTRIES = [
    "technology", "banking", "insurance", "automotive", "energy", "utilities",
    "telecommunications", "pharmaceuticals", "healthcare", "retail",
    "consumer goods", "airlines", "shipping", "manufacturing", "semiconductors",
    "mining", "real estate", "fintech", "media", "industrials",
    "agriculture/food", "construction", "chemicals", "education"
]
GEOGRAPHY_WEIGHTS = {
    "US/Canada": 20,
    "Western Europe": 15,
    "India": 12,
    "China/HK": 8,
    "Japan/Korea": 8,
    "SE Asia": 8,
    "Middle East": 7,
    "Africa": 6,
    "Latin America": 8,
    "Australia/NZ/Eastern Europe/Central Asia": 8,
}
ENTITY_PATTERNS = [
    "listed_fictional", "private_family", "subsidiary_unit", "bank_lender",
    "pe_sponsor", "state_owned", "startup_scaleup", "consortium",
    "unnamed_midcap_group", "fictional_executive"
]
OPENING_PATTERNS = [
    "number_first", "date_first", "regulator_court_agency_first",
    "verb_first_headline", "participle_phrase", "location_first",
    "sector_noun_first", "attribution_first", "company_first"
]
AMOUNT_PATTERNS = [
    "none", "small", "mid", "large", "percentage_only", "per_share_or_price",
    "multi_currency", "rating_ratio_or_date_only"
]

# Compatibility maps keep obviously nonsensical subtype-stage combinations out of batch plans.
ALLOWED_STAGES = {
    "Credit Event": {
        "missed_interest_payment": ["grace_period", "missed", "acknowledged", "creditor_negotiation"],
        "missed_principal_payment": ["grace_period", "missed", "acknowledged", "creditor_negotiation"],
        "bond_default": ["warning_signs", "grace_period", "missed", "acknowledged", "creditor_negotiation", "forbearance"],
        "loan_default": ["warning_signs", "missed", "acknowledged", "creditor_negotiation", "forbearance"],
        "bankruptcy_filing": ["warning_signs", "filed", "court_process"],
        "insolvency": ["warning_signs", "filed", "court_process"],
        "receivership_administration": ["filed", "court_process"],
        "distressed_debt_restructuring": ["creditor_negotiation", "restructuring_agreed", "resolved_after_distress"],
        "covenant_breach": ["warning_signs", "acknowledged", "creditor_negotiation", "forbearance"],
        "debt_acceleration": ["acknowledged", "creditor_negotiation", "court_process"],
        "liquidity_crisis_threatening_repayment": ["warning_signs", "creditor_negotiation", "forbearance"],
        "credit_rating_downgrade": ["warning_signs", "downgraded"],
        "negative_watch_or_outlook_credit": ["warning_signs", "watch_placed"],
        "distress_restructuring_negotiations": ["creditor_negotiation", "forbearance", "restructuring_agreed"],
        "payment_moratorium": ["acknowledged", "creditor_negotiation", "court_process"],
        "debt_standstill": ["creditor_negotiation", "forbearance", "restructuring_agreed"],
    },
    "Merger / Acquisition": {
        "acquisition": ["proposed", "announced", "signed", "pending_regulatory_review", "regulatory_approved", "completed"],
        "merger": ["considering", "proposed", "announced", "signed", "shareholder_approved", "pending_regulatory_review", "regulatory_approved", "completed"],
        "takeover": ["rumored", "considering", "proposed", "announced", "rejected", "withdrawn", "completed"],
        "takeover_bid": ["rumored", "considering", "proposed", "announced", "rejected", "withdrawn"],
        "tender_offer": ["proposed", "announced", "completed", "withdrawn"],
        "business_unit_acquisition": ["proposed", "announced", "signed", "completed"],
        "controlling_asset_acquisition": ["proposed", "announced", "signed", "completed"],
        "divestiture_business_sale": ["considering", "proposed", "announced", "signed", "completed", "terminated"],
        "merger_agreement": ["signed", "shareholder_approved", "pending_regulatory_review", "regulatory_approved", "terminated"],
        "acquisition_agreement": ["signed", "shareholder_approved", "pending_regulatory_review", "regulatory_approved", "terminated"],
        "regulatory_approval_of_deal": ["pending_regulatory_review", "regulatory_approved"],
        "shareholder_approval_of_deal": ["shareholder_approved"],
        "completed_acquisition": ["completed"],
        "rejected_takeover_offer": ["rejected"],
        "terminated_or_cancelled_deal": ["withdrawn", "terminated"],
        "rumored_or_considered_deal": ["rumored", "considering"],
    },
    "Product Launch": {
        "physical_product_launch": ["preview", "unveiling", "announcement", "pre_order", "initial_release", "commercial_availability", "market_rollout", "regional_expansion"],
        "software_product_launch": ["preview", "unveiling", "announcement", "initial_release", "commercial_availability", "market_rollout", "regional_expansion"],
        "ai_product_service_launch": ["preview", "unveiling", "announcement", "initial_release", "commercial_availability", "market_rollout"],
        "financial_product_service_launch": ["announcement", "initial_release", "commercial_availability", "market_rollout", "regional_expansion"],
        "digital_platform_launch": ["preview", "announcement", "initial_release", "commercial_availability", "market_rollout"],
        "enterprise_service_launch": ["announcement", "initial_release", "commercial_availability", "market_rollout"],
        "commercial_release": ["initial_release", "commercial_availability"],
        "public_availability": ["commercial_availability", "regional_expansion"],
        "product_unveiling": ["preview", "unveiling", "announcement"],
        "rollout": ["market_rollout", "regional_expansion"],
        "geographic_market_launch": ["market_rollout", "regional_expansion"],
        "new_service_launch": ["announcement", "initial_release", "commercial_availability", "market_rollout"],
        "delayed_launch": ["delay"],
        "cancelled_launch": ["cancellation"],
        "beta_to_commercial_launch": ["initial_release", "commercial_availability"],
        "next_generation_product_introduction": ["preview", "unveiling", "announcement", "pre_order", "initial_release"],
    },
}

def weighted_choice(weight_map):
    keys = list(weight_map.keys())
    weights = list(weight_map.values())
    return rng.choices(keys, weights=weights, k=1)[0]

def choose_stage(label, subtype, taxonomy):
    allowed = ALLOWED_STAGES[label].get(subtype)
    if allowed:
        stage_weights = taxonomy["stage_weights"]
        weights = [stage_weights.get(s, 1) for s in allowed]
        return rng.choices(allowed, weights=weights, k=1)[0]
    return weighted_choice(taxonomy["stage_weights"])

def build_batch_plan(label, count, coverage_counter=None):
    taxonomy = CLASS_TAXONOMIES[label]
    plan = []
    used_combo = set()

    for slot_i in range(count):
        # A few attempts to avoid identical combinatorial slots inside one batch.
        for _ in range(30):
            subtype = weighted_choice(taxonomy["subtype_weights"])
            stage = choose_stage(label, subtype, taxonomy)
            style = weighted_choice(STYLE_WEIGHTS)
            difficulty = weighted_choice(DIFFICULTY_WEIGHTS)
            industry = rng.choice(INDUSTRIES)
            geography = weighted_choice(GEOGRAPHY_WEIGHTS)
            scenario_hint = rng.choice(taxonomy["scenario_families"])
            entity_pattern = rng.choice(ENTITY_PATTERNS)
            opening_pattern = rng.choice(OPENING_PATTERNS)
            amount_pattern = rng.choice(AMOUNT_PATTERNS)
            combo = (subtype, stage, style, difficulty, industry, geography, scenario_hint)
            if combo not in used_combo:
                used_combo.add(combo)
                break

        plan.append({
            "slot": slot_i + 1,
            "subtype": subtype,
            "event_stage": stage,
            "style": style,
            "difficulty": difficulty,
            "industry": industry,
            "geography": geography,
            "scenario_family": scenario_hint,
            "entity_pattern": entity_pattern,
            "opening_pattern": opening_pattern,
            "amount_pattern": amount_pattern,
        })
    return plan



# ===== NOTEBOOK CELL 10 =====

# ============================================================
# Text normalization, local quality filters, and near-duplicate engine
# ============================================================

PLACEHOLDER_PATTERNS = [
    r"\[company(?: name)?\]",
    r"\[country\]",
    r"\[amount\]",
    r"\[date\]",
    r"\[product\]",
    r"<company>",
    r"<amount>",
    r"\bINSERT COMPANY\b",
    r"\bPLACEHOLDER\b",
]
META_LEAK_PATTERNS = [
    r"\bclass\s*[234]\b",
    r"\blabel\s*[:=]",
    r"\bevent\s*type\s*[:=]",
    r"^\s*credit event\s*:",
    r"^\s*m&a\s*:",
    r"^\s*product launch\s*:",
]

def normalize_text(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    return text

def first_three_words(text):
    toks = re.findall(r"[A-Za-z0-9$%'-]+", normalize_text(text))
    return " ".join(toks[:3]) if len(toks) >= 3 else " ".join(toks)

def local_validate_candidate(rec, expected_label, expected_id):
    reasons = []

    text = rec.get("text")
    if not isinstance(text, str) or not text.strip():
        reasons.append("EMPTY_TEXT")
        return reasons

    if rec.get("label") != expected_label:
        reasons.append("WRONG_LABEL")
    try:
        if int(rec.get("label_id")) != int(expected_id):
            reasons.append("WRONG_LABEL_ID")
    except Exception:
        reasons.append("INVALID_LABEL_ID")

    words = re.findall(r"\b\w+\b", text)
    if len(words) < 6:
        reasons.append("TOO_SHORT")
    if len(words) > 140:
        reasons.append("TOO_LONG")

    low = text.lower()
    if any(re.search(p, low, flags=re.I) for p in PLACEHOLDER_PATTERNS):
        reasons.append("PLACEHOLDER")
    if any(re.search(p, low, flags=re.I) for p in META_LEAK_PATTERNS):
        reasons.append("LABEL_LEAKAGE")

    required_meta = [
        "subtype", "style", "difficulty", "event_stage",
        "industry", "geography", "scenario_family"
    ]
    for key in required_meta:
        if key not in rec or rec.get(key) in (None, ""):
            reasons.append(f"MISSING_{key.upper()}")

    if rec.get("style") not in GENERATOR_STYLES:
        reasons.append("INVALID_STYLE")
    if rec.get("difficulty") not in GENERATOR_DIFFICULTIES:
        reasons.append("INVALID_DIFFICULTY")

    return reasons

# Fixed-dimensional vectorizer avoids re-fitting a vocabulary after each accepted batch.
near_dup_vectorizer = HashingVectorizer(
    n_features=2**18,
    alternate_sign=False,
    analyzer="char_wb",
    ngram_range=(3, 5),
    norm="l2",
    lowercase=True,
)

def vectorize_texts(texts):
    return near_dup_vectorizer.transform([str(t) for t in texts])

def max_similarity(candidate_texts, reference_matrix):
    if reference_matrix is None or reference_matrix.shape[0] == 0 or not candidate_texts:
        return np.zeros(len(candidate_texts), dtype=float)
    cmat = vectorize_texts(candidate_texts)
    sims = cmat @ reference_matrix.T
    if sparse.issparse(sims):
        # sparse max returns a matrix; convert to flat array
        return np.asarray(sims.max(axis=1).toarray()).ravel()
    return np.asarray(sims).max(axis=1)

def within_batch_duplicate_mask(texts, threshold=SIMILARITY_THRESHOLD):
    if len(texts) <= 1:
        return [False] * len(texts)
    mat = vectorize_texts(texts)
    sim = cosine_similarity(mat, dense_output=True)
    np.fill_diagonal(sim, 0.0)
    dup = [False] * len(texts)
    # Deterministically keep the first among near-duplicate pairs.
    for i in range(len(texts)):
        for j in range(i):
            if sim[i, j] >= threshold and not dup[j]:
                dup[i] = True
                break
    return dup



# ===== NOTEBOOK CELL 11 =====

# ============================================================
# Checkpoint discovery / state initialization
# ============================================================

ACCEPTED_CKPT = WORKDIR / "synthetic_accepted_checkpoint.parquet"
REJECTED_CKPT = WORKDIR / "synthetic_rejected_checkpoint.parquet"
CANDIDATES_CKPT = WORKDIR / "synthetic_candidates_checkpoint.parquet"
STATE_CKPT = WORKDIR / "generation_state.json"

def find_prior_checkpoint(filename):
    local = WORKDIR / filename
    if local.exists():
        return local
    hits = find_candidates(INPUT_ROOT, [filename])
    return hits[0] if hits else None

def load_df_checkpoint(filename):
    p = find_prior_checkpoint(filename)
    if p is None:
        return pd.DataFrame()
    try:
        df = pd.read_parquet(p)
        print(f"Loaded checkpoint {filename}: {len(df)} rows from {p}")
        return df
    except Exception as e:
        print(f"WARNING: Could not load checkpoint {p}: {e}")
        return pd.DataFrame()

accepted_df = load_df_checkpoint(ACCEPTED_CKPT.name)
rejected_df = load_df_checkpoint(REJECTED_CKPT.name)
candidates_df = load_df_checkpoint(CANDIDATES_CKPT.name)

state_path = find_prior_checkpoint(STATE_CKPT.name)
if state_path:
    try:
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
        print("Loaded generation state from", state_path)
    except Exception:
        state = {}
else:
    state = {}

state.setdefault("prompt_version", PROMPT_VERSION)
state.setdefault("batch_number", 0)
state.setdefault("api_calls_generator", 0)
state.setdefault("api_calls_verifier", 0)
state.setdefault("api_errors", 0)
state.setdefault("duplicate_rejections", 0)
state.setdefault("local_rejections", 0)
state.setdefault("verifier_rejections", 0)
state.setdefault("batch_plans_used", [])
state.setdefault("last_checkpoint_utc", None)

# Recompute accepted counts from checkpoint data, not from possibly stale JSON.
def accepted_counts():
    if accepted_df.empty or "label" not in accepted_df.columns:
        return {k: 0 for k in TARGETS}
    vc = accepted_df["label"].value_counts()
    return {k: int(vc.get(k, 0)) for k in TARGETS}

print("Accepted counts at start:", accepted_counts())



# ===== NOTEBOOK CELL 12 =====

# ============================================================
# Build real + accepted dedupe pool
# ============================================================

real_texts = train_real["text"].astype(str).tolist()
accepted_texts_existing = (
    accepted_df["text"].astype(str).tolist()
    if not accepted_df.empty and "text" in accepted_df.columns
    else []
)

dedupe_pool_texts = real_texts + accepted_texts_existing
dedupe_pool_matrix = vectorize_texts(dedupe_pool_texts)

exact_seen = set(normalize_text(t) for t in dedupe_pool_texts)
scenario_fingerprints = defaultdict(list)

if not accepted_df.empty and "label" in accepted_df.columns and "scenario_family" in accepted_df.columns:
    for row in accepted_df[["label", "scenario_family"]].dropna().itertuples(index=False):
        scenario_fingerprints[str(row.label)].append(str(row.scenario_family))

print("Dedupe pool rows:", len(dedupe_pool_texts))



# ===== NOTEBOOK CELL 13 =====

# ============================================================
# Prompt rendering
# ============================================================

def render_generation_prompt(label, batch_id, plan, real_refs=None):
    template = CLASS_PROMPTS[label]
    requested_count = len(plan)
    previous_fp = scenario_fingerprints[label][-60:]

    replacements = {
        "<<BATCH_ID>>": batch_id,
        "<<REQUESTED_COUNT>>": str(requested_count),
        "<<BATCH_PLAN_JSON>>": json.dumps(plan, ensure_ascii=False, indent=2),
        "<<PREVIOUS_FINGERPRINTS>>": json.dumps(previous_fp, ensure_ascii=False),
        "<<REAL_TRAIN_REFERENCES>>": json.dumps(real_refs or [], ensure_ascii=False, indent=2),
    }
    out = template
    for key, value in replacements.items():
        out = out.replace(key, value)
    return out

def render_verifier_prompt(candidate_payload):
    return VERIFIER_BATCH_PROMPT.replace(
        "<<CANDIDATES_JSON>>",
        json.dumps(candidate_payload, ensure_ascii=False, indent=2),
    )

def sample_real_train_references(label, n=3):
    if not USE_REAL_TRAIN_REFERENCES:
        return []
    pool = train_real[train_real["label"].astype(str) == label]
    if pool.empty:
        return []
    n = min(n, len(pool))
    sample = pool.sample(n=n, random_state=rng.randint(0, 10_000_000))
    return sample["text"].astype(str).tolist()



# ===== NOTEBOOK CELL 14 =====

# ============================================================
# Robust Groq calls
# ============================================================

def retry_after_seconds(exc):
    try:
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if headers:
            val = headers.get("retry-after")
            if val is not None:
                return max(float(val), 0.0)
    except Exception:
        pass
    return None

def call_with_retries(callable_fn, call_name):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = callable_fn()
            if REQUEST_DELAY_SECONDS > 0:
                time.sleep(REQUEST_DELAY_SECONDS)
            return result
        except Exception as exc:
            last_exc = exc
            state["api_errors"] = int(state.get("api_errors", 0)) + 1
            status = getattr(exc, "status_code", None)
            retry_after = retry_after_seconds(exc)

            # Respect server Retry-After when present; otherwise exponential backoff.
            sleep_for = retry_after if retry_after is not None else BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            sleep_for = min(float(sleep_for), 120.0)

            print(f"{call_name} failed (attempt {attempt}/{MAX_RETRIES}, status={status}): {type(exc).__name__}: {exc}")
            if attempt < MAX_RETRIES:
                print(f"Sleeping {sleep_for:.1f}s before retry...")
                time.sleep(sleep_for)

    raise RuntimeError(f"{call_name} failed after {MAX_RETRIES} attempts") from last_exc

def generate_batch(label, batch_id, plan):
    taxonomy = CLASS_TAXONOMIES[label]
    label_id = LABEL_TO_ID[label]
    subtypes = list(taxonomy["subtype_weights"].keys())
    stages = list(taxonomy["stage_weights"].keys())
    response_format = build_generator_response_format(label, label_id, subtypes, stages)

    real_refs = sample_real_train_references(label, REAL_TRAIN_REFERENCES_PER_CALL)
    user_prompt = render_generation_prompt(label, batch_id, plan, real_refs)

    seed = rng.randint(1, 2_000_000_000)
    temp = rng.choice([0.8, 1.0, 1.1])

    def _call():
        return client.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": GLOBAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
            reasoning_effort=GENERATOR_REASONING_EFFORT,
            include_reasoning=False,
            temperature=temp,
            top_p=0.95,
            max_completion_tokens=GENERATOR_MAX_COMPLETION_TOKENS,
            seed=seed,
        )

    response = call_with_retries(_call, f"generator:{label}")
    state["api_calls_generator"] += 1

    content = response.choices[0].message.content
    data = json.loads(content)

    if data.get("batch_id") != batch_id:
        raise ValueError(f"Generator batch_id mismatch: expected {batch_id}, got {data.get('batch_id')}")

    examples = data.get("examples", [])
    if len(examples) != len(plan):
        raise ValueError(f"Generator returned {len(examples)} examples; expected {len(plan)}")

    return examples

def verify_candidates(records):
    if not records:
        return []

    payload = [
        {
            "candidate_id": r["candidate_id"],
            "text": r["text"],
            "intended_label": r["label"],
            "intended_label_id": int(r["label_id"]),
        }
        for r in records
    ]
    prompt = render_verifier_prompt(payload)

    def _call():
        return client.chat.completions.create(
            model=VERIFIER_MODEL,
            messages=[
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=VERIFIER_RESPONSE_FORMAT,
            reasoning_effort=VERIFIER_REASONING_EFFORT,
            include_reasoning=False,
            temperature=0.1,
            top_p=1.0,
            max_completion_tokens=VERIFIER_MAX_COMPLETION_TOKENS,
        )

    response = call_with_retries(_call, "verifier")
    state["api_calls_verifier"] += 1

    data = json.loads(response.choices[0].message.content)
    results = data.get("results", [])
    expected_ids = [r["candidate_id"] for r in records]
    got_ids = [r.get("candidate_id") for r in results]

    if got_ids != expected_ids:
        raise ValueError(
            "Verifier candidate IDs/order mismatch. "
            f"Expected {expected_ids}, got {got_ids}"
        )

    return results



# ===== NOTEBOOK CELL 15 =====

# ============================================================
# Provenance helpers + checkpoint save
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def safe_concat(base_df, rows):
    if not rows:
        return base_df
    add_df = pd.DataFrame(rows)
    if base_df is None or base_df.empty:
        return add_df.reset_index(drop=True)
    return pd.concat([base_df, add_df], ignore_index=True, sort=False)

def checkpoint():
    global accepted_df, rejected_df, candidates_df

    if accepted_df is not None and not accepted_df.empty:
        accepted_df.to_parquet(ACCEPTED_CKPT, index=False)
    else:
        pd.DataFrame().to_parquet(ACCEPTED_CKPT, index=False)

    if rejected_df is not None and not rejected_df.empty:
        rejected_df.to_parquet(REJECTED_CKPT, index=False)
    else:
        pd.DataFrame().to_parquet(REJECTED_CKPT, index=False)

    if candidates_df is not None and not candidates_df.empty:
        candidates_df.to_parquet(CANDIDATES_CKPT, index=False)
    else:
        pd.DataFrame().to_parquet(CANDIDATES_CKPT, index=False)

    state["accepted_counts"] = accepted_counts()
    state["last_checkpoint_utc"] = utc_now()
    STATE_CKPT.write_text(json.dumps(state, indent=2), encoding="utf-8")

def append_to_dedupe_pool(new_texts):
    global dedupe_pool_matrix, dedupe_pool_texts, exact_seen
    if not new_texts:
        return
    mat = vectorize_texts(new_texts)
    dedupe_pool_matrix = sparse.vstack([dedupe_pool_matrix, mat], format="csr")
    dedupe_pool_texts.extend(new_texts)
    exact_seen.update(normalize_text(t) for t in new_texts)

def synthetic_record(candidate, batch_id, verifier_result):
    out = dict(candidate)
    out.update({
        "synthetic_id": candidate["candidate_id"],
        "source_dataset": "groq_synthetic",
        "source_label": candidate["label"],
        "source_subtype": candidate.get("subtype"),
        "source_split": "synthetic_train",
        "source_id": candidate["candidate_id"],
        "mapping_quality": "synthetic_verified",
        "generator_model": GENERATOR_MODEL,
        "verifier_model": VERIFIER_MODEL,
        "batch_id": batch_id,
        "prompt_version": PROMPT_VERSION,
        "verifier_confidence": float(verifier_result["confidence"]),
        "verifier_predicted_label": verifier_result["predicted_label"],
        "verifier_predicted_label_id": int(verifier_result["predicted_label_id"]),
        "verifier_reason_code": verifier_result["reason_code"],
        "accepted_utc": utc_now(),
    })
    return out



# ===== NOTEBOOK CELL 16 =====

# ============================================================
# One full generation -> local filter -> dedupe -> verifier cycle
# ============================================================

def process_one_batch(label, requested_count):
    global accepted_df, rejected_df, candidates_df

    state["batch_number"] += 1
    batch_id = f"{label.lower().replace(' ', '_').replace('/', '_')}_{state['batch_number']:06d}"

    plan = build_batch_plan(label, requested_count)
    state["batch_plans_used"].append({
        "batch_id": batch_id,
        "label": label,
        "requested_count": requested_count,
    })
    # Keep JSON state compact.
    if len(state["batch_plans_used"]) > 300:
        state["batch_plans_used"] = state["batch_plans_used"][-300:]

    generated = generate_batch(label, batch_id, plan)

    prepared = []
    local_rejected = []
    for i, rec in enumerate(generated):
        rec = dict(rec)
        rec["candidate_id"] = f"{batch_id}_c{i+1:03d}"
        rec["batch_id"] = batch_id
        rec["generated_utc"] = utc_now()

        reasons = local_validate_candidate(rec, label, LABEL_TO_ID[label])
        norm = normalize_text(rec.get("text", ""))

        if not reasons and norm in exact_seen:
            reasons.append("EXACT_DUPLICATE")

        if reasons:
            rej = dict(rec)
            rej["rejection_stage"] = "local"
            rej["rejection_reason"] = "|".join(reasons)
            local_rejected.append(rej)
        else:
            prepared.append(rec)

    state["local_rejections"] += len(local_rejected)

    # Near-duplicate check against REAL TRAIN + previously accepted synthetic.
    if prepared:
        texts = [r["text"] for r in prepared]
        sim_to_pool = max_similarity(texts, dedupe_pool_matrix)
        within_dup = within_batch_duplicate_mask(texts)

        survivors = []
        for rec, sim, is_within_dup in zip(prepared, sim_to_pool, within_dup):
            if float(sim) >= SIMILARITY_THRESHOLD or is_within_dup:
                rej = dict(rec)
                rej["rejection_stage"] = "dedupe_pre_verifier"
                rej["rejection_reason"] = (
                    "NEAR_DUPLICATE_WITH_BATCH"
                    if is_within_dup else
                    f"NEAR_DUPLICATE_WITH_POOL:{float(sim):.4f}"
                )
                local_rejected.append(rej)
                state["duplicate_rejections"] += 1
            else:
                survivors.append(rec)
        prepared = survivors

    # Store all raw candidates + local status.
    raw_rows = []
    rejected_ids = {r["candidate_id"] for r in local_rejected}
    for rec in generated:
        pass
    # We have candidate IDs only in local_rejected/prepared, so combine those records.
    raw_rows.extend([{**r, "candidate_status": "local_rejected"} for r in local_rejected])
    raw_rows.extend([{**r, "candidate_status": "sent_to_verifier"} for r in prepared])
    candidates_df = safe_concat(candidates_df, raw_rows)
    rejected_df = safe_concat(rejected_df, local_rejected)

    # Independent verification in small batches.
    verified_accept = []
    verifier_rejected = []

    for start in range(0, len(prepared), VERIFICATION_BATCH_SIZE):
        chunk = prepared[start:start + VERIFICATION_BATCH_SIZE]
        results = verify_candidates(chunk)

        for rec, vr in zip(chunk, results):
            intended_label = rec["label"]
            intended_id = int(rec["label_id"])

            final_accept = (
                vr["accept"] is True
                and vr["predicted_label"] == intended_label
                and int(vr["predicted_label_id"]) == intended_id
                and float(vr["confidence"]) >= VERIFIER_CONFIDENCE_THRESHOLD
                and vr["ambiguity"] is False
                and vr["reason_code"] == "PASS"
            )

            if final_accept:
                verified_accept.append(synthetic_record(rec, batch_id, vr))
            else:
                rej = dict(rec)
                rej.update({
                    "rejection_stage": "verifier",
                    "rejection_reason": vr["reason_code"],
                    "verifier_predicted_label": vr["predicted_label"],
                    "verifier_predicted_label_id": int(vr["predicted_label_id"]),
                    "verifier_confidence": float(vr["confidence"]),
                    "verifier_ambiguity": bool(vr["ambiguity"]),
                })
                verifier_rejected.append(rej)

    state["verifier_rejections"] += len(verifier_rejected)

    # Second dedupe pass after verifier acceptance, sequentially against the pool
    # so two verifier-passed candidates cannot both enter if they are near duplicates.
    final_accept_rows = []
    post_verify_reject = []

    for rec in verified_accept:
        norm = normalize_text(rec["text"])
        if norm in exact_seen:
            rr = dict(rec)
            rr["rejection_stage"] = "dedupe_post_verifier"
            rr["rejection_reason"] = "EXACT_DUPLICATE"
            post_verify_reject.append(rr)
            state["duplicate_rejections"] += 1
            continue

        sim = float(max_similarity([rec["text"]], dedupe_pool_matrix)[0])
        if sim >= SIMILARITY_THRESHOLD:
            rr = dict(rec)
            rr["rejection_stage"] = "dedupe_post_verifier"
            rr["rejection_reason"] = f"NEAR_DUPLICATE_WITH_POOL:{sim:.4f}"
            post_verify_reject.append(rr)
            state["duplicate_rejections"] += 1
            continue

        final_accept_rows.append(rec)
        append_to_dedupe_pool([rec["text"]])
        scenario_fingerprints[label].append(str(rec.get("scenario_family", "")))

    rejected_df = safe_concat(rejected_df, verifier_rejected + post_verify_reject)
    accepted_df = safe_concat(accepted_df, final_accept_rows)

    checkpoint()

    return {
        "batch_id": batch_id,
        "generated": len(generated),
        "local_or_dedupe_rejected": len(local_rejected),
        "sent_to_verifier": len(prepared),
        "verifier_rejected": len(verifier_rejected),
        "post_verify_dedupe_rejected": len(post_verify_reject),
        "accepted": len(final_accept_rows),
    }



# ===== NOTEBOOK CELL 17 =====

# ============================================================
# Main generation loop
# ============================================================

def target_for_run(label):
    full_target = TARGETS[label]
    if PILOT_MODE:
        return min(full_target, PILOT_ACCEPTED_PER_CLASS)
    return full_target

def run_generation():
    total_batches_this_run = 0

    for label in ["Credit Event", "Merger / Acquisition", "Product Launch"]:
        target = target_for_run(label)
        empty_streak = 0
        class_batches = 0

        while accepted_counts()[label] < target:
            if class_batches >= MAX_BATCHES_PER_CLASS_PER_RUN:
                print(f"Reached MAX_BATCHES_PER_CLASS_PER_RUN for {label}. Saving and moving on.")
                break
            if total_batches_this_run >= MAX_TOTAL_BATCHES_PER_RUN:
                print("Reached MAX_TOTAL_BATCHES_PER_RUN. Saving and stopping.")
                checkpoint()
                return

            remaining = target - accepted_counts()[label]
            request_n = min(GENERATION_BATCH_SIZE, remaining)

            try:
                stats = process_one_batch(label, request_n)
                class_batches += 1
                total_batches_this_run += 1

                print(
                    f"[{stats['batch_id']}] generated={stats['generated']} "
                    f"sent_to_verifier={stats['sent_to_verifier']} "
                    f"accepted={stats['accepted']} | "
                    f"{label}: {accepted_counts()[label]}/{target}"
                )

                if stats["accepted"] == 0:
                    empty_streak += 1
                else:
                    empty_streak = 0

                if empty_streak >= MAX_CONSECUTIVE_EMPTY_BATCHES:
                    print(
                        f"Stopping {label}: {MAX_CONSECUTIVE_EMPTY_BATCHES} consecutive "
                        "batches accepted zero examples. Review prompts/rejections before continuing."
                    )
                    break

            except KeyboardInterrupt:
                print("Interrupted by user. Saving checkpoints.")
                checkpoint()
                return
            except Exception as exc:
                print(f"\nBatch failed for {label}: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=2)
                checkpoint()
                print("Progress saved. Stopping this run so quota/API/schema issues can be inspected safely.")
                return

    checkpoint()

if RUN_GENERATION:
    run_generation()
else:
    print("RUN_GENERATION=False: no API generation executed.")

print("\nAccepted counts after run:", accepted_counts())



# ===== NOTEBOOK CELL 18 =====

# ============================================================
# Save final synthetic outputs + augmented training dataset
# ============================================================

def save_dual(df, stem):
    pq = WORKDIR / f"{stem}.parquet"
    csv = WORKDIR / f"{stem}.csv"
    df.to_parquet(pq, index=False)
    df.to_csv(csv, index=False)
    print("Saved:", pq)
    print("Saved:", csv)

# Ensure empty dataframes still have useful outputs.
accepted_final = accepted_df.copy() if accepted_df is not None else pd.DataFrame()
rejected_final = rejected_df.copy() if rejected_df is not None else pd.DataFrame()

for label, stem in [
    ("Credit Event", "synthetic_credit_event"),
    ("Merger / Acquisition", "synthetic_merger_acquisition"),
    ("Product Launch", "synthetic_product_launch"),
]:
    part = (
        accepted_final[accepted_final["label"] == label].copy()
        if not accepted_final.empty and "label" in accepted_final.columns
        else pd.DataFrame()
    )
    save_dual(part, stem)

save_dual(accepted_final, "synthetic_accepted_all")
save_dual(rejected_final, "synthetic_rejected_all")

# Align synthetic columns to real columns while preserving both real provenance and synthetic metadata.
if accepted_final.empty:
    event_train_augmented = train_real.copy()
else:
    real_aligned = train_real.copy()
    syn_aligned = accepted_final.copy()
    union_cols = list(dict.fromkeys(list(real_aligned.columns) + list(syn_aligned.columns)))
    real_aligned = real_aligned.reindex(columns=union_cols)
    syn_aligned = syn_aligned.reindex(columns=union_cols)
    event_train_augmented = pd.concat([real_aligned, syn_aligned], ignore_index=True, sort=False)

save_dual(event_train_augmented, "event_train_augmented")

if val_real is not None:
    save_dual(val_real, "event_validation_real_frozen")
if test_real is not None:
    save_dual(test_real, "event_test_real_frozen")



# ===== NOTEBOOK CELL 19 =====

# ============================================================
# Final assertions
# ============================================================

# 1-2. Taxonomy and mapping
assert LABEL_TO_ID == {
    "Geopolitical": 0,
    "Macroeconomic": 1,
    "Credit Event": 2,
    "Merger / Acquisition": 3,
    "Product Launch": 4,
}

if not accepted_final.empty:
    # 3-4-5. Synthetic only for IDs 2/3/4; no Geo/Macro.
    assert set(accepted_final["label"].unique()).issubset(SYNTHETIC_LABELS)
    assert set(accepted_final["label_id"].astype(int).unique()).issubset({2, 3, 4})
    assert not accepted_final["label"].isin(["Geopolitical", "Macroeconomic"]).any()

    # 6. No null/empty text.
    assert not accepted_final["text"].isna().any()
    assert not (accepted_final["text"].astype(str).str.strip() == "").any()

    # 7-8. No exact synthetic-vs-real-train duplicates or accepted synthetic duplicates.
    real_norm = set(train_real["text"].astype(str).map(normalize_text))
    syn_norm = accepted_final["text"].astype(str).map(normalize_text)
    assert not any(t in real_norm for t in syn_norm)
    assert not syn_norm.duplicated().any()

    # 9. Every accepted row passed verifier.
    assert (accepted_final["verifier_reason_code"] == "PASS").all()
    assert (accepted_final["verifier_confidence"].astype(float) >= VERIFIER_CONFIDENCE_THRESHOLD).all()
    assert (
        accepted_final["verifier_predicted_label"].astype(str)
        == accepted_final["label"].astype(str)
    ).all()
    assert (
        accepted_final["verifier_predicted_label_id"].astype(int)
        == accepted_final["label_id"].astype(int)
    ).all()

# 10-11. Validation/test are untouched real inputs; we never append synthetic to them.
if val_real is not None and "source_dataset" in val_real.columns:
    assert not val_real["source_dataset"].astype(str).eq("groq_synthetic").any()
if test_real is not None and "source_dataset" in test_real.columns:
    assert not test_real["source_dataset"].astype(str).eq("groq_synthetic").any()

# 12. Augmented training contains every real-train row plus accepted synthetic rows.
assert len(event_train_augmented) == len(train_real) + len(accepted_final)

print("ALL FINAL ASSERTIONS PASSED.")



# ===== NOTEBOOK CELL 20 =====

# ============================================================
# Report + plots
# ============================================================

real_counts = train_real["label"].value_counts().to_dict()
accepted_counts_final = (
    accepted_final["label"].value_counts().to_dict()
    if not accepted_final.empty and "label" in accepted_final.columns else {}
)
final_train_counts = event_train_augmented["label"].value_counts().to_dict()

rejection_reason_counts = {}
if not rejected_final.empty and "rejection_reason" in rejected_final.columns:
    rejection_reason_counts = rejected_final["rejection_reason"].astype(str).value_counts().to_dict()

report = {
    "created_utc": utc_now(),
    "prompt_version": PROMPT_VERSION,
    "generator_model": GENERATOR_MODEL,
    "verifier_model": VERIFIER_MODEL,
    "real_train_counts": {str(k): int(v) for k, v in real_counts.items()},
    "requested_accepted_targets": TARGETS,
    "accepted_synthetic_counts": {str(k): int(v) for k, v in accepted_counts_final.items()},
    "total_accepted_synthetic": int(len(accepted_final)),
    "total_rejected_synthetic": int(len(rejected_final)),
    "rejection_reason_counts": {str(k): int(v) for k, v in rejection_reason_counts.items()},
    "final_augmented_train_counts": {str(k): int(v) for k, v in final_train_counts.items()},
    "api_calls_generator": int(state.get("api_calls_generator", 0)),
    "api_calls_verifier": int(state.get("api_calls_verifier", 0)),
    "api_errors": int(state.get("api_errors", 0)),
    "local_rejections": int(state.get("local_rejections", 0)),
    "verifier_rejections": int(state.get("verifier_rejections", 0)),
    "duplicate_rejections": int(state.get("duplicate_rejections", 0)),
    "complete_targets_met": {
        label: int(accepted_counts_final.get(label, 0)) >= int(target)
        for label, target in TARGETS.items()
    },
}

for col in ["subtype", "style", "difficulty", "event_stage", "industry", "geography"]:
    if not accepted_final.empty and col in accepted_final.columns:
        report[f"{col}_distribution"] = {
            str(k): int(v) for k, v in accepted_final[col].value_counts().to_dict().items()
        }

report_path = WORKDIR / "synthetic_generation_report.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("Saved:", report_path)
print(json.dumps(report["complete_targets_met"], indent=2))

# Plots
def plot_counts(series, title, filename):
    ax = series.plot(kind="bar", figsize=(9, 4))
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Rows")
    plt.tight_layout()
    out = WORKDIR / filename
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()
    print("Saved:", out)

plot_counts(
    train_real["label"].value_counts().reindex(LABEL_TO_ID.keys(), fill_value=0),
    "REAL training class distribution",
    "plot_real_train_distribution.png",
)

plot_counts(
    event_train_augmented["label"].value_counts().reindex(LABEL_TO_ID.keys(), fill_value=0),
    "Augmented training class distribution",
    "plot_augmented_train_distribution.png",
)

if not accepted_final.empty:
    plot_counts(
        accepted_final["label"].value_counts().reindex(TARGETS.keys(), fill_value=0),
        "Accepted synthetic examples",
        "plot_synthetic_accepted.png",
    )



# ===== NOTEBOOK CELL 21 =====

# ============================================================
# Manual quality audit samples
# ============================================================

if accepted_final.empty:
    print("No accepted synthetic examples yet.")
else:
    for label in ["Credit Event", "Merger / Acquisition", "Product Launch"]:
        part = accepted_final[accepted_final["label"] == label]
        print(f"\n{'='*80}\n{label}: random accepted sample\n{'='*80}")
        if part.empty:
            print("No accepted examples.")
        else:
            cols = [c for c in [
                "text", "subtype", "style", "difficulty", "event_stage",
                "industry", "geography", "verifier_confidence"
            ] if c in part.columns]
            display(part[cols].sample(n=min(30, len(part)), random_state=RANDOM_SEED))

if not rejected_final.empty:
    print(f"\n{'='*80}\nRejected sample\n{'='*80}")
    cols = [c for c in [
        "text", "label", "rejection_stage", "rejection_reason",
        "verifier_predicted_label", "verifier_confidence"
    ] if c in rejected_final.columns]
    display(rejected_final[cols].sample(n=min(20, len(rejected_final)), random_state=RANDOM_SEED))

