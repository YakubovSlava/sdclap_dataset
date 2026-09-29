# Appendix — Full prompt lists used for downstream evaluation (R2, item 2)

All lists were composed by hand by the authors (not LLM-generated), in the same style as the
`description` field the model was trained on. For Dusha, the phrases were written specifically to
match the operational class definitions from the corpus's own assessor guideline [Kondratenko et
al., 2022, arXiv 2212.12266, §3.3]. The 15 descriptions per class are averaged after projection
into the joint embedding space (prompt ensembling, as in CLIP).

For the external comparison models (LAION-CLAP, CLAP, CLSP, ParaCLAP, ParaSpeechCLAP — unified
format) on RESD/Dusha, English 15-synonym lists are used (see below), substituted into a single
shared template, `"spoken in a {synonym} tone"`.

Additionally, to check the robustness of the comparison to the choice of prompt format, the same
15-synonym lists were substituted into the ORIGINAL template of each model, matching its source
publication (verified directly against the papers and the authors' official code): LAION-CLAP —
`"This is a sound of {synonym}"` (Wu et al., 2023, §4.3); CLAP — `"this person is feeling
{synonym}"` (Elizalde et al., 2022, §3.3, the emotion-specific template); CLSP — `"A speaker in a
{synonym} tone."` (Yang et al., 2026). The direction of the comparison (SDCLAP leading on
RESD/Dusha) does not change when the unified format is replaced with each model's own original
template (see `для ревью/table4_official_prompts_recheck.md`). For ParaCLAP, the original format
consists of acoustic descriptors (pitch/loudness/arousal-valence, not emotion words, e.g. `"has a
high pitch"`, `"speaker is calm"`) — incompatible by type with direct emotion-word classification,
so it was not re-evaluated with its original format.

---

## RESD_PROMPTS — for SDCLAP on RESD (Russian, 7 classes × 15)

**anger**: злобно крикнула; говорил со злостью; раздраженно; гневно закричал; яростно воскликнул;
сердито сказал; со злобой процедил; рявкнул; рассерженно бросил; злобно прошипел; в бешенстве
закричал; недовольно проворчал; огрызнулся; вспылил и крикнул; зло бросил

**disgust**: говорил с отвращением; брезгливо сказал; с презрением произнёс; гадливо поморщился;
с омерзением сказал; презрительно бросил; скривившись, сказал; с гримасой отвращения произнёс;
пренебрежительно фыркнул; с брезгливостью процедил; отвращённо прошептал; с гадливостью сказал;
свысока произнёс; холодно и с презрением сказал; скептически поморщился

**enthusiasm**: говорил с воодушевлением; восторженно воскликнул; с энтузиазмом сказал;
взволнованно и радостно произнёс; с жаром сказал; вдохновенно произнёс; увлечённо рассказывал;
с азартом воскликнул; пылко произнёс; живо откликнулся; загорелся и сказал; эмоционально
воскликнул; с придыханием сказал; восхищённо произнёс; горячо поддержал

**fear**: говорил испуганно; дрожащим голосом произнёс; со страхом прошептал; перепуганно
закричал; в ужасе воскликнул; с тревогой сказал; трясущимся голосом произнёс; испуганно
пробормотал; в панике закричал; со страхом в голосе сказал; боязливо прошептал; дрожа, произнёс;
с ужасом прошептал; нервно и испуганно сказал; охваченный страхом, пробормотал

**happiness**: говорил радостно; весело сказал; смеялся; счастливо воскликнул; с улыбкой
произнёс; радостно рассмеялся; довольно улыбнулся и сказал; ликующе воскликнул; со смехом
произнёс; радостно хихикнул; просиял и сказал; с восторгом рассмеялся; игриво произнёс;
беззаботно рассмеялся; счастливо улыбнулся

**neutral**: говорил спокойно; ровным голосом сказал; без эмоций произнёс; буднично сказал;
невозмутимо произнёс; спокойно заметил; сдержанно сказал; бесстрастно произнёс; равнодушно
сказал; монотонно произнёс; деловито сказал; флегматично заметил; просто сказал; хладнокровно
произнёс; обыденно заметил

**sadness**: плача; печально сказал; грустно произнёс; со слезами в голосе сказал; уныло произнёс;
со вздохом сказал; подавленно произнёс; горестно вздохнул и сказал; тоскливо сказал; всхлипывая,
произнёс; с грустью в голосе сказал; убитым голосом произнёс; рыдая, сказал; скорбно произнёс;
печально вздохнул

---

## DUSHA_PROMPTS — for SDCLAP on Dusha (Russian, 4 classes × 15, written specifically to match the
assessor guideline)

**positive** (official definition: the text is spoken with a smile, laughter, admiration, a
playful tone, OR positive words are intonationally emphasized): говорил с улыбкой; со смехом
произнёс; радостно улыбаясь сказал; с восхищением сказал; игриво произнёс; весело сказал; с
восторгом воскликнул; довольным тоном сказал; радостно рассмеялся; с улыбкой в голосе произнёс;
игривым тоном сказал; с одобрением сказал; жизнерадостно сказал; с теплотой в голосе сказал;
довольно усмехнулся

**neutral** (the voice is calm and even, WITHOUT EMOTION — even if the text itself is positive or
negative): говорил ровным голосом; без всякой эмоции сказал; монотонно произнёс; бесстрастно
сказал; сухо и ровно произнёс; без интонации сказал; невозмутимо сказал; спокойным ровным тоном
произнёс; буднично, без эмоций сказал; флегматично произнёс; сдержанно, без выражения сказал;
механически произнёс; с каменным лицом сказал; равнодушным тоном произнёс; ровно, ничего не
выражая сказал

**sad** (sadness, longing, a FADING voice): печально сказал; с тоской в голосе произнёс; угасающим
голосом сказал; грустно, тихо произнёс; с грустью в голосе сказал; подавленным тоном сказал;
уныло произнёс; тоскливо сказал; печально вздохнул; с надрывом в голосе сказал; потухшим голосом
произнёс; меланхолично произнёс; со скорбью в голосе сказал; тихим угасающим голосом сказал; со
вздохом грусти сказал

**angry** (anger or irritation, OR shouting, OR speaking through clenched teeth, OR intonationally
emphasized negative words — quiet/restrained anger is deliberately included, not shouting alone):
закричал от злости; заорал в гневе; сказал сквозь стиснутые зубы; процедил сквозь зубы; рявкнул
раздражённо; сказал с плохо скрываемым раздражением; гневно повысил голос; сдержанно, но зло
сказал; прошипел сквозь зубы; раздражённо бросил; сказал с нажимом и злостью; выкрикнул в
ярости; сказал резким, злым тоном; процедил с раздражением; сказал, едва сдерживая гнев

---

## SITUATIONAL_PROMPTS — for SDCLAP on the ParaSpeech benchmark (Russian, 21 classes × 15)

**angry**: злобно крикнула; говорил со злостью; раздраженно; гневно закричал; яростно
воскликнул; сердито сказал; со злобой процедил; рявкнул; рассерженно бросил; злобно прошипел; в
бешенстве закричал; недовольно проворчал; огрызнулся; вспылил и крикнул; зло бросил

**guilt**: виновато произнёс; с чувством вины сказал; виновато пробормотал; смущённо извинился;
виноватым тоном сказал; стыдливо признал; с раскаянием произнёс; виновато опустив глаза сказал;
покаянно сказал; виновато прошептал; неловко извинился; с сожалением признался; смущённо
промямлил; виновато оправдывался; с угрызениями совести сказал

**scared**: говорил испуганно; дрожащим голосом произнёс; со страхом прошептал; перепуганно
закричал; в ужасе воскликнул; с тревогой сказал; трясущимся голосом произнёс; испуганно
пробормотал; в панике закричал; со страхом в голосе сказал; боязливо прошептал; дрожа, произнёс;
с ужасом прошептал; нервно и испуганно сказал; охваченный страхом, пробормотал

**happy**: говорил радостно; весело сказал; смеялся; счастливо воскликнул; с улыбкой произнёс;
радостно рассмеялся; довольно улыбнулся и сказал; ликующе воскликнул; со смехом произнёс;
радостно хихикнул; просиял и сказал; с восторгом рассмеялся; игриво произнёс; беззаботно
рассмеялся; счастливо улыбнулся

**loud**: громко крикнул; во весь голос произнёс; громогласно заявил; прокричал во весь голос;
громко выкрикнул; оглушительно крикнул; громким голосом сказал; гаркнул; прогремел на весь зал;
громко объявил; зычным голосом крикнул; во всю силу лёгких закричал; громогласно провозгласил;
трубным голосом произнёс; оглушительно объявил

**sarcastic**: саркастически произнёс; с сарказмом сказал; язвительно заметил; насмешливо
процедил; с издёвкой сказал; иронично усмехнулся и сказал; ехидно заметил; с плохо скрытой
издёвкой сказал; саркастически хмыкнул; язвительно бросил; с насмешкой в голосе сказал; колко
заметил; с сарказмом протянул; издевательски произнёс; с притворным восхищением сказал

**sympathetic**: сочувственно сказал; с сочувствием произнёс; участливо сказал; мягко и с
состраданием произнёс; сочувственно вздохнул и сказал; с теплотой и пониманием сказал;
утешительно произнёс; сострадательно сказал; с искренним сочувствием произнёс; участливо
спросил; заботливо сказал; с сочувствием в голосе произнёс; по-доброму утешил; сердечно
посочувствовал; с пониманием и теплом сказал

**desirous**: с желанием в голосе сказал; жаждуще произнёс; мечтательно сказал; с нескрываемым
желанием произнёс; томно сказал; страстно произнёс; с вожделением сказал; жадно произнёс; с
тоской желания сказал; нетерпеливо желая произнёс; с придыханием желания сказал; алчно произнёс;
с горящими глазами сказал; жаждущим тоном произнёс; с плохо скрываемым желанием сказал

**enthusiastic**: говорил с воодушевлением; восторженно воскликнул; с энтузиазмом сказал;
взволнованно и радостно произнёс; с жаром сказал; вдохновенно произнёс; увлечённо рассказывал; с
азартом воскликнул; пылко произнёс; живо откликнулся; загорелся и сказал; эмоционально
воскликнул; с придыханием сказал; восхищённо произнёс; горячо поддержал

**saddened**: печально сказал; грустно произнёс; со слезами в голосе сказал; уныло произнёс; со
вздохом сказал; подавленно произнёс; горестно вздохнул и сказал; тоскливо сказал; всхлипывая,
произнёс; с грустью в голосе сказал; убитым голосом произнёс; рыдая, сказал; скорбно произнёс;
печально вздохнул; плача

**anxious**: тревожно сказал; с беспокойством произнёс; нервно сказал; взволнованно произнёс; с
тревогой в голосе сказал; обеспокоенно спросил; нервничая, произнёс; с волнением сказал;
встревоженно произнёс; беспокойно пробормотал; с плохо скрываемым волнением сказал; напряжённо
произнёс; неспокойно сказал; с тревожной ноткой произнёс; взвинченно сказал

**sleepy**: сонно пробормотал; сонным голосом сказал; зевая, произнёс; заспанно сказал; вялым
сонным голосом произнёс; полусонно пробормотал; сонно протянул; устало и сонно сказал; зевнул и
сказал; сонным, тягучим голосом произнёс; клюя носом, пробормотал; разморенно сказал; сонно и
лениво произнёс; с трудом продирая глаза, сказал; дремотно пробормотал

**admiring**: восхищённо сказал; с восхищением произнёс; восторженно произнёс; с нескрываемым
восхищением сказал; изумлённо восхитился; почтительно произнёс; с уважением и восторгом сказал;
восхищённо покачал головой и сказал; с благоговением произнёс; с восторгом воскликнул;
восхищённо прошептал; с трепетом произнёс; поражённо и с восхищением сказал; любуясь, сказал; с
восхищённым придыханием произнёс

**disgusted**: говорил с отвращением; брезгливо сказал; с презрением произнёс; гадливо
поморщился; с омерзением сказал; презрительно бросил; скривившись, сказал; с гримасой отвращения
произнёс; пренебрежительно фыркнул; с брезгливостью процедил; отвращённо прошептал; с
гадливостью сказал; свысока произнёс; холодно и с презрением сказал; скептически поморщился

**awed**: благоговейно произнёс; с трепетом сказал; поражённо прошептал; изумлённо сказал; с
благоговейным ужасом произнёс; потрясённо пробормотал; затаив дыхание, произнёс; с благоговением
прошептал; поражённый увиденным, сказал; оцепенело произнёс; с немым восторгом сказал; изумлённо
выдохнул; заворожённо прошептал; с трепетным восторгом произнёс; потрясённый, едва произнёс

**pained**: произнёс сквозь боль; простонал; болезненно произнёс; с гримасой боли сказал;
морщась от боли, сказал; мучительно произнёс; со стоном сказал; сдавленно от боли произнёс;
превозмогая боль, сказал; прохрипел от боли; с болью в голосе произнёс; простонав, сказал;
страдальчески произнёс; стиснув зубы от боли, сказал; болезненно поморщившись, произнёс

**fast**: быстро протараторил; скороговоркой произнёс; торопливо сказал; быстро выпалил;
затараторил; спешно произнёс; быстрой скороговоркой сказал; торопясь, произнёс; выпалил на одном
дыхании; быстро и сбивчиво произнёс; тараторя, сказал; поспешно пробормотал; быстро протарахтел;
скороговоркой выпалил; торопливо протараторил

**calm**: говорил спокойно; ровным голосом сказал; без эмоций произнёс; буднично сказал;
невозмутимо произнёс; спокойно заметил; сдержанно сказал; бесстрастно произнёс; равнодушно
сказал; монотонно произнёс; деловито сказал; флегматично заметил; просто сказал; хладнокровно
произнёс; обыденно заметил

**whispered**: прошептал; еле слышно прошептал; шёпотом сказал; тихим шёпотом произнёс; едва
слышно прошептал; прошептал на ухо; чуть слышно прошептал; шёпотом произнёс; тихо, почти шёпотом
сказал; прошелестел шёпотом; приглушённым шёпотом произнёс; прошептал одними губами; тихим
шёпотом выдохнул; почти беззвучно прошептал; заговорщицким шёпотом сказал

**enunciated**: чётко произнёс; внятно и раздельно сказал; чеканя каждое слово, произнёс;
отчётливо произнёс; тщательно выговаривая слова, сказал; чётко и раздельно произнёс; внятно
проговорил; с чёткой дикцией произнёс; медленно и отчётливо произнёс; чеканным голосом произнёс;
выговаривая каждый слог, сказал; ясно и чётко произнёс; раздельно, по слогам произнёс; тщательно
артикулируя, сказал; подчёркнуто чётко произнёс

**confused**: растерянно сказал; смущённо произнёс; недоумённо спросил; озадаченно произнёс;
растерянно пробормотал; в замешательстве сказал; сбивчиво и растерянно произнёс; недоумевая,
спросил; растерянно развёл руками и сказал; озадаченно почесал затылок и сказал; смущённо и
растерянно пробормотал; в полном недоумении произнёс; растерянно моргнул и сказал; сбитый с
толку, спросил; недоумённо пожал плечами и сказал

---

## RESD_SYNONYMS_EN — for the external models on RESD (English, 7 classes × 15)

**anger**: angry, furious, irate, enraged, mad, wrathful, incensed, livid, cross, indignant,
irritated, infuriated, outraged, fuming, vexed

**disgust**: disgusted, repulsed, revolted, repelled, nauseated, sickened, appalled, disdainful,
contemptuous, scornful, averse, repugnant, distasteful, loathing, abhorrent

**enthusiasm**: enthusiastic, eager, excited, passionate, zealous, keen, fervent, avid, animated,
spirited, ardent, energetic, exuberant, thrilled, elated

**fear**: afraid, scared, frightened, terrified, fearful, anxious, alarmed, panicked,
apprehensive, timid, nervous, dreadful, spooked, petrified, uneasy

**happiness**: happy, joyful, cheerful, glad, delighted, pleased, elated, content, jubilant,
merry, blissful, gleeful, upbeat, buoyant, satisfied

**neutral**: neutral, calm, flat, even, unemotional, plain, matter-of-fact, dispassionate,
detached, indifferent, level, composed, impassive, stoic, monotone

**sadness**: sad, sorrowful, mournful, melancholy, gloomy, dejected, downcast, unhappy, doleful,
despondent, wistful, forlorn, blue, heavy-hearted, woeful

---

## DUSHA_SYNONYMS_EN — for the external models on Dusha (English, 4 classes × 15)

**positive**: positive, cheerful, upbeat, joyful, happy, warm, friendly, pleasant, delighted,
glad, buoyant, sunny, bright, playful, affectionate

**neutral**: neutral, calm, even, flat, unemotional, plain, dispassionate, matter-of-fact,
detached, composed, level, impassive, monotone, indifferent, stoic

**sad**: sad, sorrowful, mournful, gloomy, downcast, dejected, melancholy, doleful, despondent,
forlorn, blue, unhappy, wistful, heavy-hearted, woeful

**angry**: angry, irritated, furious, annoyed, cross, mad, indignant, incensed, irate,
exasperated, resentful, testy, riled, fuming, vexed

---

## PARASPEECH_SYNONYMS_15 — for the external models on ParaSpeech (English, 21 classes × 15)

**angry**: furious, irate, enraged, mad, wrathful, incensed, livid, cross, indignant, irritated,
infuriated, outraged, fuming, vexed, exasperated

**guilt**: remorseful, guilty, contrite, regretful, ashamed, penitent, sorry, apologetic, rueful,
self-reproachful, shamefaced, conscience-stricken, chastened, abashed, repentant

**scared**: frightened, afraid, terrified, fearful, alarmed, panicked, apprehensive, timid,
nervous, dreadful, spooked, petrified, uneasy, startled, jumpy

**happy**: joyful, cheerful, glad, delighted, pleased, elated, content, jubilant, merry, blissful,
gleeful, upbeat, buoyant, satisfied, thrilled

**loud**: booming, noisy, thunderous, blaring, deafening, resounding, roaring, clamorous,
vociferous, piercing, raucous, strident, bellowing, earsplitting, sonorous

**sarcastic**: mocking, sardonic, derisive, snide, cutting, scornful, ironic, caustic, biting,
wry, taunting, contemptuous, acerbic, dry, satirical

**sympathetic**: compassionate, empathetic, caring, understanding, kind, gentle, supportive,
consoling, tender, soothing, warm, considerate, solicitous, comforting, benevolent

**desirous**: longing, yearning, craving, wistful, eager, wanting, hungry, covetous, lustful,
aching, pining, hankering, avid, thirsty, wishful

**enthusiastic**: eager, excited, passionate, zealous, keen, fervent, avid, animated, spirited,
ardent, energetic, exuberant, thrilled, gung-ho, enthused

**saddened**: sorrowful, mournful, melancholy, gloomy, dejected, downcast, unhappy, doleful,
despondent, wistful, forlorn, blue, heavy-hearted, woeful, grieving

**anxious**: nervous, worried, uneasy, apprehensive, tense, jittery, restless, fretful, on-edge,
agitated, troubled, distressed, edgy, wound-up, unsettled

**sleepy**: drowsy, tired, weary, groggy, dozy, lethargic, somnolent, sluggish, yawning, listless,
heavy-eyed, torpid, languid, nodding-off, half-asleep

**admiring**: adoring, appreciative, approving, impressed, reverent, awed, esteeming, worshipful,
respectful, enamored, captivated, fond, starstruck, venerating, admiration-filled

**disgusted**: repulsed, revolted, repelled, nauseated, sickened, appalled, disdainful,
contemptuous, scornful, averse, repugnant, distasteful, loathing, abhorrent, grossed-out

**awed**: astonished, amazed, astounded, awestruck, stunned, wonderstruck, dumbfounded,
marveling, spellbound, overwhelmed, reverent, flabbergasted, staggered, impressed, breathless

**pained**: anguished, hurting, suffering, aching, distressed, agonized, tormented, wounded,
afflicted, sore, hurt, smarting, tortured, grieved, stricken

**fast**: rapid, quick, hurried, brisk, swift, hasty, speedy, rushed, breakneck, fleet,
accelerated, prompt, expeditious, nimble, brisk-paced

**calm**: relaxed, serene, tranquil, peaceful, composed, placid, unruffled, cool, collected,
steady, even-tempered, restful, quiet, settled, at-ease

**whispered**: murmured, muttered, breathed, hushed, faint, soft-spoken, low, undertoned, quiet,
susurrous, muffled, barely-audible, subdued, confidential, hoarse-whispered

**enunciated**: articulated, pronounced, voiced, clearly-spoken, crisp, precise, distinct,
well-formed, sharp, carefully-spoken, clipped, deliberate, measured, exact, clean-spoken

**confused**: bewildered, puzzled, perplexed, baffled, disoriented, muddled, flustered,
mystified, befuddled, confounded, uncertain, dazed, stumped, addled, at-a-loss
