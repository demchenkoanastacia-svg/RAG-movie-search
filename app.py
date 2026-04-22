"""
RAG Knowledge Base - Starter Template
FAMNIT AI Course - Day 3

A simple Retrieval-Augmented Generation (RAG) app built with
Streamlit, LangChain, and ChromaDB. No API keys needed!

Instructions:
  1. Replace the DOCUMENTS list below with your own texts
  2. Update the app title and description
  3. Run locally:  streamlit run app.py
  4. Deploy to Render (see assignment instructions)
"""

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="My **MOVIE** RAG Knowledge Base",
    page_icon="🔍🎬",
    layout="wide",
)

# --- Custom Styling for Aesthetics ---


st.markdown("""
<style>
/* Main App Background and Sidebar */
[data-testid="stAppViewContainer"] {
    background-color: #0f0f0f;
}
[data-testid="stSidebar"] {
    background-color: #141414;
    border-right: 1px solid #E50914; /* Movie Red Accent */
}

/* Hide Radio Button Dots completely */
div[role="radiogroup"] > label > div:first-of-type {
    display: none !important;
}

/* Style the Radio List Items (Sidebar Navigation) */
div[role="radiogroup"] > label {
    padding: 10px 15px;
    border-radius: 6px;
    margin-bottom: 5px;
    background-color: transparent;
    transition: all 0.2s ease;
    cursor: pointer;
}

/* Hover effect */
div[role="radiogroup"] > label:hover {
    background-color: rgba(229, 9, 20, 0.15); /* subtle red */
}

/* Highlight Selected Page (Red Background) */
div[role="radiogroup"] > label[data-checked="true"] {
    background-color: #E50914 !important; 
}
div[role="radiogroup"] > label[data-checked="true"] p {
    color: white !important;
    font-weight: bold;
}

/* Metrics and Containers styled darkly */
div[data-testid="stMetric"] {
    background-color: #1a1a1a;
    border-left: 4px solid #E50914;
    padding: 15px;
    border-radius: 8px;
}
div[data-testid="stExpander"], div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #151515;
    border-color: #333333;
}

/* General Text color optimizations for dark mode */
h1, h2, h3, p, span, div.stMarkdown, div.stText, label {
    color: #f1f1f1 !important;
}

/* Input fields */
.stTextInput > div > div > input {
    color: #ffffff !important;
    border-color: #E50914 !important;
}

/* Dividers */
hr {
    border-bottom-color: #E50914 !important;
}


</style>
""", unsafe_allow_html=True)




# ──────────────────────────────────────────────────────────────────────
# YOUR DOCUMENTS — Replace these with your own topic!
# Each string is one "document" that will be chunked, embedded, and
# stored in the vector database for semantic search.
# ──────────────────────────────────────────────────────────────────────
DOCUMENTS = [
    {"title": "Mad Max 2", "plot": "After a global war results in widespread oil shortages and ecocide, civilisation collapses and the world descends into barbarism. Former policeman Max Rockatansky, haunted by the death of his family, drives around the desert Outback of what was once Australia, scavenging for food and petrol with his dog. He outmanoeuvres a group of marauders led by biker Wez using his driving skills and a shotgun. He steals petrol from the wrecked vehicles of one of his pursuers and inspects a wrecked semi-trailer and prime mover.\nLater, Max tries collecting an apparently abandoned gyrocopter's fuel, but is ambushed by the pilot. Max overpowers the man with his dog's help, sparing his life in return for being led to a working oil refinery the pilot has discovered. They arrive during the daily attack on the facility by a motorised gang, of which Wez is a member.\nThe next day, Max witnesses cars leave the besieged compound and get chased down by marauders. He witnesses a man, Nathan, get attacked by the gang and once he becomes incapacitated, they rip the clothes off the woman companion and kill her. After, Max rescues Nathan, now the sole survivor of the car, and strikes a deal to return him to the complex in exchange for fuel, but the man dies after Max gets him back, and the leader of the settlers, Pappagallo, says the deal died with Nathan. The settlers are about to confiscate Max's car and cast him out of their compound when the marauders return to negotiate. A feral child who lives in the refinery compound kills Wez's partner with a metal boomerang and Wez wants revenge, but the gang's leader, a masked man called \"Lord Humungus\", offers to spare the settlers' lives in exchange for their fuel supply and leaves for the day. However, the settlers are divided over whether or not they can trust Humungus.\nMax offers his own deal: he will bring them the semi-truck he saw earlier so they can try to haul away their tanker full of oil, if they return his car and give him as much fuel as he can carry. The settlers agree, and that night Max sneaks past the marauders on foot carrying fuel for the truck. He encounters the Gyro Captain and forces him carry the gasoline to the truck, which he gets started. It is somewhat damaged as Max passes through the marauders' encampment en route to the refinery, but he makes it, followed by the gyrocopter.\nMax refuses Pappagallo's entreaty to accompany the settlers to a fabled northern paradise, opting instead to collect his fuel and leave. Wez catches him using Humungus's nitrous oxide-equipped vehicle and causes him to crash. A Marauder kills Max's dog and is about to kill the seriously injured Max when Marauder Toadie attempts to siphon the fuel from the tanks of Max's car, triggering the vehicle to self-destruct. Left for dead, Max is rescued by the Gyro Captain and returned to the compound.\nDespite his injuries, Max insists on driving the repaired truck during the escape. His support consists of the Gyro Captain, Pappagallo in a separate vehicle, three of the settlers on the outside of the armoured tanker, and the Feral Kid, who jumps on the truck as it is leaving. The marauders pursue the tanker, allowing the remaining settlers to flee their compound in a caravan of smaller vehicles after rigging the refinery to explode.\nPappagallo and the three settlers are killed and the Gyro Captain is shot down. Max turns the truck around and, as he is fighting with Wez, Humungus collides with the truck head on, killing Wez and himself. The truck rolls off the road and the surviving marauders survey the scene, only to abandon their chase when they see the tanker leaking sand and not gas. As Max carries the Feral Kid from the wrecked tanker, he inspects the sand pouring out. The Gyro Captain drives up and the two share a grin as Max realises the tanker was a diversion the whole time. They rendezvous with the settlers, who transported the fuel in oil drums inside their vehicles.\nThe Gyro Captain succeeds Pappagallo as leader of the settlers and takes them north. The grown Feral Kid, \"Chief of the Great Northern Tribe,\" reveals in voice-over that he never saw \"the Road Warrior\" again.", "genre": "Action", "poster": "https://upload.wikimedia.org/wikipedia/en/f/f7/Mad_max_two_the_road_warrior.jpg"},
    {"title": "Die Hard", "plot": "On Christmas Eve, New York City Police Department (NYPD) Detective John McClane arrives in Los Angeles, hoping to reconcile with his estranged wife, Holly, at a party held by her employer, the Nakatomi Corporation. He is driven to Nakatomi Plaza by a limo driver, Argyle, who offers to wait for McClane in the garage. While McClane washes himself, the tower is seized by German ex-radical Hans Gruber and his heavily armed team, including Karl and Theo. Everyone in the tower is taken hostage except for McClane, who slips away, and Argyle, who remains oblivious to events.\nGruber is posing as a terrorist to steal the $640 million  in untraceable bearer bonds in the building's vault. He kills executive Joseph Takagi after failing to extract the access code from him and tasks Theo with breaking into the vault. The terrorists are alerted to McClane's presence, and Karl's brother, Tony, is sent after him. McClane kills Tony and takes his weapon and radio, which he uses to contact the skeptical Los Angeles Police Department (LAPD). Sergeant Al Powell is sent to investigate. Meanwhile, McClane kills more terrorists and recovers their bag of C-4 and detonators. Realizing Powell is about to leave, having found nothing amiss, McClane drops a terrorist's corpse onto his car. After Powell calls for backup, a SWAT team attempts to storm the building but is counterattacked by the terrorists. McClane throws some C-4 down an elevator shaft, causing an explosion that kills some of the terrorists and ends the counterattack.\nHolly's co-worker Harry Ellis attempts to negotiate on Gruber's behalf but is killed by Gruber when McClane refuses to surrender. While checking the explosives on the roof, Gruber encounters McClane and pretends to be an escaped hostage; McClane gives Gruber a gun. Gruber attempts to shoot McClane but finds the weapon is unloaded, and he is saved only by the intervention of other terrorists. McClane escapes but is injured by shattered glass and loses the detonators. Outside, Federal Bureau of Investigation (FBI) agents take control. They order the power to be shut off, which, as Gruber had anticipated, disables the final vault lock so his team can collect the bonds.\nThe FBI agrees to Gruber's demand for a helicopter, intending to send helicopter gunships to eliminate the group. McClane realizes Gruber plans to blow the roof to kill the hostages and fake his team's deaths. Karl, enraged by Tony's death, attacks McClane and is seemingly killed. Gruber sees a news report by Richard Thornburg on McClane's children and infers that he is Holly's husband. The hostages are taken to the roof while Gruber keeps Holly with him. McClane drives the hostages from the roof just before Gruber detonates it and destroys the approaching FBI helicopters. Meanwhile, Theo retrieves an escape vehicle from the parking garage but is knocked out by Argyle, who has been following events on the limo's CB radio.\nA weary and battered McClane finds Holly with Gruber and his remaining henchman. McClane seemingly surrenders to Gruber and is about to be shot but grabs his concealed service pistol taped to his back and uses his last two bullets to wound Gruber and kill his accomplice. Gruber crashes through a window but grabs onto Holly's wristwatch and makes a last-ditch attempt to kill the pair. McClane unclasps the watch, and Gruber falls to his death. Outside, Karl ambushes McClane and Holly, only to be shot dead by Powell. Holly punches Thornburg when he attempts to interview McClane. Argyle crashes through the parking garage door in the limo and drives McClane and Holly away together.", "genre": "Action, Christmas", "poster": "https://upload.wikimedia.org/wikipedia/en/c/ca/Die_Hard_%281988_film%29_poster.jpg"},
    {"title": "Pinocchio", "plot": "In a sleepy village in Italy, Jiminy Cricket arrives at the shop of a woodworker and toymaker named Geppetto, who creates a puppet he names Pinocchio. As he falls asleep, Geppetto wishes upon a star for Pinocchio to be a real boy. Late that night, the Blue Fairy visits the workshop and brings Pinocchio to life, although he remains a puppet. She informs him that if he proves himself to be brave, truthful, and unselfish, he will become a real boy. When Jiminy reveals himself, the Blue Fairy assigns him to be Pinocchio's conscience. Geppetto awakens upon hearing the commotion from Pinocchio falling, and is overjoyed to discover that he is alive and will become a real boy.\nThe next morning, while walking to school, Pinocchio is led astray by con artist fox Honest John and his sidekick Gideon the Cat. Honest John convinces him to join Stromboli's puppet show, despite Jiminy's protestations. Pinocchio becomes Stromboli's star attraction, but when he tries to go home, Stromboli locks him in a bird cage and leaves to tour the world with Pinocchio. After Jiminy unsuccessfully tries to free his friend, the Blue Fairy appears, and an anxious Pinocchio lies about what happened, causing his nose to grow and become a tree branch with a bird's nest. The Blue Fairy restores his nose and frees Pinocchio when he promises to make amends, but warns him that she can offer no further help.\nMeanwhile, a mysterious Coachman hires Honest John to find disobedient and naughty boys for him to take to Pleasure Island, a notorious and infamous place. Honest John, despite the legal risks and the Coachman's implication of what happens to the boys, accepts the job out of fear, and finds Pinocchio, persuading him to take a vacation on Pleasure Island. On the way to the island, Pinocchio befriends Lampwick, a delinquent boy. At Pleasure Island, without rules or authority to enforce their activity, Pinocchio, Lampwick, and many other boys soon engage in vices such as vandalism, fighting, smoking and drinking. Jiminy eventually finds Pinocchio in a bar smoking and playing pool with Lampwick, and the two have a falling out after Pinocchio defends Lampwick for his actions. As Jiminy tries leaving Pleasure Island, he discovers that the island hides a horrible curse that transforms the boys into donkeys after making \"jackasses\" of themselves, and they are sold by the Coachman into slave labor. Pinocchio witnesses Lampwick transform into a donkey, and with Jiminy's help, he flees before he can be fully transformed himself, though he still has a donkey's ears and tail.\nUpon returning home, Pinocchio and Jiminy find Geppetto's workshop deserted, and obtain a letter from the Blue Fairy in the form of a dove, stating that Geppetto had set out to sea in search for Pinocchio on Pleasure Island, but got swallowed by a gigantic and vicious sperm whale called Monstro and is now trapped in its belly. Determined to rescue his father, Pinocchio jumps into the Mediterranean Sea with Jiminy and is soon swallowed by Monstro, where he reunites with Geppetto. Pinocchio devises a scheme to make Monstro sneeze and allow them to escape, but the whale chases them and destroys their raft with his tail. Pinocchio selflessly pulls Geppetto to safety in a cove just as Monstro crashes into it and Pinocchio is killed in the process.\nBack at home, Geppetto, Jiminy, Figaro, and Cleo mourn Pinocchio. Having succeeded in proving himself brave, truthful, and unselfish, Pinocchio is revived and turned into a real human boy by the Blue Fairy, much to everyone's joy. As the group celebrates, Jiminy steps outside to thank the Fairy and is rewarded with a solid gold badge that certifies him as an official conscience.", "genre": "Animation", "poster": "https://upload.wikimedia.org/wikipedia/en/b/ba/Pinocchio-1940-poster.jpg"},
    {"title": "Toy Story", "plot": "A group of sentient toys, led by Sheriff Woody, are preparing to move to a new house with their young owner, Andy Davis and his family. During the party for Andy's sixth birthday, the toys—including Mr. Potato Head, Slinky Dog, Rex the tyrannosaur, Hamm the piggy bank, and Bo Peep the porcelain doll—become concerned that Andy might receive a new toy that will replace them. Andy receives a Buzz Lightyear action figure, who believes he is an actual Space Ranger and does not know he is a toy. Buzz impresses the others with his electronic features and becomes Andy's new favorite toy, provoking Woody's jealousy.\nTwo days before the move, Andy's family is planning to have dinner at a restaurant called Pizza Planet. To ensure Andy brings him along and not Buzz, Woody tries to knock Buzz behind a desk, but accidentally knocks him out the window instead. Andy takes Woody into the car, where Buzz furiously confronts him. The two fight, fall out of the car, and are left behind, but then manage to hitch a ride on a Pizza Planet delivery truck. At Pizza Planet, Buzz mistakes a claw machine arcade game for a rocket and climbs in, pursued by Woody. Sid Phillips, Andy's sadistic next-door neighbor, grabs Woody and Buzz with the claw and takes them to his house, where they encounter his mutant toys, made from parts of toys Sid has destroyed.\nBuzz sees a television commercial promoting Buzz Lightyear toys and suffers an existential crisis, finally realizing he is a toy after all. He attempts to fly, but falls and breaks his arm. After Sid's toys repair Buzz, Sid tapes Buzz to a firework rocket, planning to blow him up the following day. Overnight, Woody helps Buzz understand that his purpose is to make Andy happy, which reinvigorates Buzz. As Sid prepares to launch the rocket, Woody and the mutant toys come to life and frighten him.\nAfter Sid flees, Woody and Buzz pursue the Davis's moving truck, but Sid's dog attacks Woody. While Buzz fights off the dog, Woody climbs into the truck, then pushes a remote-controlled car out to rescue Buzz. The other toys think Woody is harming the car, and throw him off the truck. Woody and Buzz race after the truck on the car, but its batteries run out. Woody ignites the rocket strapped to Buzz, and Buzz opens his wings to sever the tape just before the rocket explodes. Woody and Buzz glide through the sunroof of the Davis's car, landing safely inside. At Christmas in the new house, Andy receives a puppy, which causes Woody and Buzz to nervously smile at each other.", "genre": "Animation", "poster": "https://upload.wikimedia.org/wikipedia/en/1/13/Toy_Story.jpg"},
    {"title": "The Texas Chain Saw Massacre", "plot": "In the early hours of August 18, 1973, a grave robber steals several corpses from a cemetery near Newt, Muerto County, Texas. The robber ties a rotting corpse and other body parts onto a monument, creating a grisly display that is discovered by a local resident as the sun rises.\nDriving in a van, five teenagers take a road trip through the area: Sally Hardesty, Jerry, Pam, Kirk, and Sally's paraplegic brother Franklin. They stop at the cemetery to check on the grave of Sally and Franklin's grandfather, which appears undisturbed. As the group drives past a slaughterhouse, Franklin recounts the Hardesty family's history with animal slaughter. They soon pick up a hitchhiker, who talks about his family who worked at the old slaughterhouse. He borrows Franklin's pocket knife and cuts himself, then takes a single Polaroid picture of the group, for which he demands money. When they refuse to pay, he burns the photo and attacks Franklin with a straight razor. The group forces him out of the van, where he smears blood on the side as they drive off. Low on gas, the group stops at a gas station but its proprietor says that fuel is unavailable. The group explores a nearby abandoned house, owned by the Hardesty family.\nKirk and Pam leave the others behind, planning to visit a nearby swimming hole mentioned by Franklin. On their way there, they discover another house, surrounded by run-down cars, and run by gas-powered generators. Hoping to barter for gas, Kirk enters the house through the unlocked door, while Pam waits outside. As he searches the house, a large man wearing a mask made of skin appears and murders Kirk with a hammer. When Pam enters the house, she stumbles into a room strewn with decaying remains and furniture made from human and animal bones. She attempts to flee but is caught by the man and impaled on a meat hook. The man then starts up a chainsaw, dismembering Kirk as Pam watches. In the evening, Jerry searches for Pam and Kirk. When he enters the other house, he finds Pam's nearly-dead, spasming body in a chest freezer and is killed by the masked man.\nWith darkness falling, Sally and Franklin set out to find their friends. En route, the masked man ambushes them, killing Franklin with the chainsaw. The man chases Sally into the house, where she finds a very old, seemingly dead man and a woman's rotting corpse. She escapes from the man by jumping through a second-floor window, and she flees to the gas station. With the man in pursuit, Sally arrives at the gas station when he seems to disappear. The station's proprietor comforts Sally with the offer of help, after which he beats and subdues her, loading her into his pickup truck. The proprietor drives to the other house, and the hitchhiker appears. The proprietor scolds him for his actions at the cemetery, identifying the hitchhiker as the grave robber. As they enter the house, the masked man reappears, dressed in women's clothing. The proprietor identifies the masked man and the hitchhiker as brothers, and the hitchhiker refers to the masked man as \"Leatherface\". The two brothers bring the old man—\"Grandpa\"—down the stairs and cut Sally's finger so that Grandpa can suck her blood, Sally then faints from the ordeal.\nThe next morning, Sally regains consciousness. The men taunt her and bicker with each other, resolving to kill her with a hammer. They try to include Grandpa in the activity, but Grandpa is too weak. Sally breaks free and runs onto a road in front of the house, pursued by the brothers. An oncoming truck accidentally runs over the hitchhiker, killing him. The truck driver attacks Leatherface with a large wrench, causing him to fall and injure his leg with the chainsaw. The truck driver flees  while Sally, covered in blood, flags down a passing pickup truck and climbs into the bed, narrowly escaping Leatherface. As the pickup drives away, Sally laughs hysterically while an enraged Leatherface swings his chainsaw in the road as the sun rises.", "genre": "Horror", "poster": "https://upload.wikimedia.org/wikipedia/en/a/a0/The_Texas_Chain_Saw_Massacre_%281974%29_theatrical_poster.jpg"},
    {"title": "The Rocky Horror Picture Show", "plot": "The film begins with a pair of floating disembodied lips welcoming the audience to a science fiction double feature (\"Science Fiction/Double Feature\"). Throughout the film, a criminologist from an unspecified point in the future narrates and provides commentary on the events.\nFollowing the wedding of their friends, a naïve young couple, Brad Majors and Janet Weiss, get engaged and decide to celebrate with their high school science teacher Dr. Scott, who taught the class where they first met (\"Dammit Janet\"). En route to Scott's house on a dark and rainy night, they get lost and suffer a flat tyre. Seeking a telephone to call for help, the couple walks to a nearby castle (\"Over at the Frankenstein Place\") where a party is being held. They are accepted in by the strangely dressed inhabitants, led by the butler Riff Raff, the maid Magenta, and a groupie named Columbia, who dance to \"The Time Warp\". Dr. Frank-N-Furter, a transvestite mad scientist, introduces himself and invites them to stay for the night (\"Sweet Transvestite\").\nWith the help of Riff Raff, Frank brings to life a tall, muscular, handsome blond man named Rocky (\"The Sword of Damocles\"). As Frank vows he can improve Rocky into an ideal man in a week (\"I Can Make You a Man\"), Eddie, a motorcyclist with a bandaged head, breaks out of a deep freeze (\"Hot Patootie – Bless My Soul\"). Frank kills Eddie with an ice axe, justifying it as a \"mercy killing\". Rocky and Frank depart for the bridal suite (\"I Can Make You a Man (Reprise)\").\nBrad and Janet are shown to separate bedrooms, where Frank visits and seduces each one disguised as the other. Meanwhile, Riff Raff torments Rocky, who flees the suite. Janet, having learned of Brad's dalliance with Frank, discovers Rocky cowering in his birth tank. While tending to his wounds, Janet seduces Rocky as Magenta and Columbia watch from their bedroom monitor (\"Touch-a, Touch-a, Touch-a, Touch Me\").\nDr. Scott, now a government investigator of UFOs, comes to the castle in search of his nephew Eddie, who sent him a letter implying part of his brain was removed by aliens. Everyone discovers Janet and Rocky together, enraging Frank. Magenta summons everyone to an uncomfortable dinner, which they soon realise has been prepared from Eddie's mutilated remains (\"Eddie\"). In the chaos, Janet runs screaming into Rocky's arms, provoking a jealous Frank to chase her through the halls to the lab, where he uses his Medusa Transducer to turn Dr. Scott, Brad, Janet, Rocky, and Columbia into nude statues (\"Planet Schmanet Janet/Wise Up Janet Weiss\"/\"Planet Hotdog\").\nAfter dressing the statues in cabaret costumes, Frank \"unfreezes\" them and leads them in a live cabaret floor show, complete with an RKO tower and a swimming pool (\"Rose Tint My World\"/\"Don't Dream It, Be It\"/\"Wild and Untamed Thing\"). Riff Raff and Magenta interrupt and announce that due to Frank's extravagance, they are declaring mutiny and returning to their home planet of Transsexual, Transylvania. Frank makes a desperate final plea (\"I'm Going Home\"), but is ignored as Riff Raff kills both him and Columbia with a laser. An enraged Rocky climbs the tower with Frank's body, and, after several shots from the laser, plunges to his death in the pool. The castle lifts off into space, and Brad, Janet, and Dr. Scott are left crawling in the smog and dirt, confused and disorientated, as the criminologist concludes that the human race is equivalent to insects crawling on the planet's surface: \"lost in time, and lost in space ... and meaning\" (\"Super Heroes\").", "genre": "Musical", "poster": "https://upload.wikimedia.org/wikipedia/en/c/c2/Original_Rocky_Horror_Picture_Show_poster.jpg"},
    {"title": "2001: A Space Odyssey", "plot": "In a prehistoric veld, a tribe of hominins is driven away from a water hole by a rival tribe, and the next day finds an alien monolith. The tribe learns how to use the bones of dead animals as weapons and, after a successful first hunt, uses them to drive away the rival tribe. Millions of years later, Dr Heywood Floyd, Chairman of the United States National Council of Astronautics, travels to Clavius Base, an American lunar outpost. During a stopover at Space Station Five, he meets Russian scientists who are concerned that Clavius seems to be unresponsive. He refuses to discuss rumours of an epidemic at the base. At Clavius, Floyd addresses a meeting of personnel, stressing the need for secrecy regarding their newest discovery. His mission is to investigate a recently found artefact, a monolith buried four million years earlier near the lunar crater Tycho. As Floyd and others examine and photograph the object, it emits a high-powered radio signal.\nEighteen months later, the American spacecraft Discovery One is bound for Jupiter, with mission pilots and scientists Dr Dave Bowman and Dr Frank Poole on board, along with three other scientists in suspended animation. Most of Discovery's operations are controlled by HAL, a HAL 9000 computer with a human-like personality. When HAL reports the imminent failure of an antenna control device, Bowman retrieves it in an extravehicular activity (EVA) pod, but finds nothing wrong. HAL suggests reinstalling the device and letting it fail so the problem can be verified. Mission Control advises the astronauts that results from their backup 9000 computer indicate that HAL has made an error, but HAL blames it on human error. Concerned about HAL's behaviour, Bowman and Poole enter an EVA pod so they can talk in private without HAL overhearing. They agree to disconnect HAL if he is proven wrong. HAL follows their conversation by lip reading.\nWhile Poole is floating away from his pod to replace the antenna unit, HAL takes control of the pod and attacks him, sending Poole tumbling away from the ship with a severed air line. Bowman takes another pod to rescue Poole. While he is outside, HAL turns off the life support functions of the crewmen in suspended animation, killing them. When Bowman returns to the ship with Poole's body, HAL refuses to let him back in, stating that their plan to deactivate him jeopardises the mission. Bowman releases Poole's body and opens the ship's emergency airlock with his remote manipulators. Lacking a helmet for his spacesuit, he positions his pod carefully so that when he jettisons the pod's door, he is propelled by the escaping air across the vacuum into Discovery's airlock. He enters HAL's processor core and begins disconnecting HAL's memory, ignoring HAL's pleas to stop. When he is finished, a prerecorded video by Heywood Floyd plays, revealing that the mission's actual objective is to investigate the radio signal sent from the monolith to Jupiter.\nAt Jupiter, Bowman finds a third, much larger monolith orbiting the planet. He leaves Discovery in an EVA pod to investigate. He is pulled into a vortex of coloured light and observes bizarre astronomical phenomena and strange landscapes of unusual colours as he passes by. Finally, he finds himself in a large neoclassical bedroom where he sees, then becomes, older versions of himself: first standing in the bedroom, middle-aged and still in his spacesuit, then dressed in leisure attire and eating dinner, and finally as an old man lying in bed. A monolith appears at the foot of the bed, and as Bowman reaches for it, he is transformed into a foetus enclosed in a transparent orb of light, which afterwards floats in space above the Earth.", "genre": "Sci-Fi", "poster": "https://upload.wikimedia.org/wikipedia/en/1/11/2001_A_Space_Odyssey_%281968%29.png"},
    {"title": "Blade Runner", "plot": "In 2019 Los Angeles, former police officer Rick Deckard is detained by Officer Gaff, who likes to make origami figures, and is brought to his former supervisor, Bryant. Deckard, whose job as a \"blade runner\" was to track down bioengineered humanoids known as replicants and terminally \"retire\" them, is informed that four replicants are on Earth illegally. Deckard begins to leave, but Bryant makes veiled threats and Deckard stays. The two watch a video of a blade runner named Holden administering the Voight-Kampff test, which is designed to distinguish replicants from humans based on their emotional responses to questions. The test subject, Leon, shoots Holden on the second question. Bryant wants Deckard to retire Leon and three other Nexus-6 replicants: Roy Batty, Zhora, and Pris.\nBryant has Deckard meet with the CEO of the company that creates the replicants, Eldon Tyrell, so he can administer the V-K test on a Nexus-6 to see if it works on them. Tyrell expresses his interest in seeing the test fail first and asks him to administer it on his assistant Rachael. After a much longer than standard test, Deckard concludes privately to Tyrell that Rachael is a replicant who believes she is human. Tyrell explains that she is an experiment who has been given false memories to provide an \"emotional cushion\", and that she has no knowledge of her true nature.\nIn searching Leon's hotel room, Deckard finds photos and a scale from the skin of an animal, which is later identified as a synthetic snake scale. Deckard returns to his apartment, where Rachael is waiting. She tries to prove her humanity by showing him a family photo, but Deckard reveals that her memories are implants from Tyrell's niece, and she leaves in tears.\nReplicants Roy and Leon meanwhile investigate a replicant eye-manufacturing laboratory and learn of J. F. Sebastian, a gifted genetic designer who works closely with Tyrell. Pris locates Sebastian and manipulates him to gain his trust.\nA photograph from Leon's apartment and the snake scale lead Deckard to a strip club, where Zhora works. After a confrontation and chase, Deckard kills Zhora. Bryant also orders him to retire Rachael, who has disappeared from the Tyrell Corporation. Deckard spots Rachael in a crowd, but he is ambushed by Leon, who knocks the gun out of Deckard's hand and beats him. As Leon is about to kill Deckard, Rachael saves him by using Deckard's gun to kill Leon. They return to Deckard's apartment and, during a discussion, he promises not to track her down. As Rachael abruptly tries to leave, Deckard restrains her and forces her to kiss him, and she ultimately relents. Deckard leaves Rachael at his apartment and departs to search for the remaining replicants.\nRoy arrives at Sebastian's apartment and tells Pris that the other replicants are dead. Sebastian reveals that because of a genetic premature aging disorder, his life will be cut short, like the replicants that were built with a four-year lifespan. Roy uses Sebastian to gain entrance to Tyrell's penthouse. He demands more life from his maker, which Tyrell says is impossible. Roy confesses that he has done \"questionable things\" but Tyrell dismisses this, praising Roy's advanced design and accomplishments in his short life. Roy kisses Tyrell and then kills him by crushing his eyes and skull. Sebastian tries to flee and is later reported dead.\nAt Sebastian's apartment, Deckard is ambushed by Pris, but he kills her as Roy returns. Roy's body begins to fail as the end of his lifespan nears. He chases Deckard through the building and onto the roof. Deckard tries to jump onto another roof but is left hanging from the edge. Roy makes the jump with ease and, as Deckard's grip loosens, Roy hoists him onto the roof to save him. Before Roy dies, he laments that his memories \"will be lost in time, like tears in rain\". Gaff arrives to congratulate Deckard, also reminding him that Rachael will not live, but \"then again, who does?\" Deckard returns to his apartment to retrieve Rachael. While escorting her to the elevator, he notices a small origami unicorn on the floor. He recalls Gaff's words and departs with Rachael.", "genre": "Sci-Fi", "poster": "https://upload.wikimedia.org/wikipedia/en/9/9f/Blade_Runner_%281982_poster%29.png"},
    {"title": "The Dark Knight", "plot": "A gang of masked criminals rob a mafia-owned bank in Gotham City, betraying and killing each other until the sole survivor, the Joker, reveals himself as the mastermind and escapes with the money. The vigilante Batman, district attorney Harvey Dent, and police lieutenant Jim Gordon ally to eliminate Gotham's organized crime. Batman's true identity, the billionaire Bruce Wayne, publicly supports Dent as Gotham's legitimate protector, believing Dent's success will allow him to retire as Batman and romantically pursue his childhood friend Rachel Dawes—despite her being with Dent.\nGotham's mafia bosses gather to discuss protecting their organizations from the Joker, the police, and Batman. The Joker interrupts the meeting and offers to kill Batman for half of the fortune their accountant, Lau, concealed before fleeing to Hong Kong to avoid extradition. With the help of Wayne Enterprises CEO Lucius Fox, Batman finds Lau in Hong Kong and returns him to the custody of the Gotham police. His testimony enables Dent to apprehend the crime families. The bosses accept the Joker's offer, and he kills high-profile targets involved in the trial, including the judge and police commissioner. Although Gordon saves the mayor, the Joker threatens that his attacks will continue until Batman reveals his identity. He targets Dent at a fundraising dinner and throws Rachel out of a window, but Batman rescues her.\nBruce struggles to understand the Joker's motives, to which his butler Alfred Pennyworth says that \"some men just want to watch the world burn.\" Dent claims he is Batman to lure the Joker out, who attacks the police convoy transporting Dent. Batman and Gordon apprehend the Joker, and Gordon is promoted to commissioner. At the police station, Batman interrogates the Joker, who says he finds Batman entertaining and has no intention of killing him. Having deduced Batman's feelings for Rachel, the Joker reveals she and Dent are being held separately in buildings rigged to explode. Batman races to rescue Rachel while Gordon and the other officers go after Dent, but they discover the Joker has given their positions in reverse. The explosives detonate, killing Rachel and severely burning Dent's face on one side. The Joker escapes custody, extracts the fortune's location from Lau, and burns it, killing Lau in the process.\nColeman Reese, a consultant for Wayne Enterprises, deduces and tries to expose Batman's identity, but the Joker threatens to blow up a hospital unless Reese is killed. While the police evacuate hospitals and Gordon struggles to keep Reese alive, the Joker meets with a disillusioned Dent, persuading him to take the law into his own hands and avenge Rachel. Dent defers his decision-making to his now half-scarred, two-headed coin, killing the corrupt officers and the mafia involved in Rachel's death. As panic grips the city, the Joker reveals that two evacuation ferries, one carrying civilians and the other prisoners, are rigged to explode at midnight unless one group sacrifices the other. To the Joker's disbelief, the passengers refuse to kill one another. Batman subdues the Joker but refuses to kill him. Before the police arrest the Joker, he says that although Batman proved incorruptible, his plan to corrupt Dent has succeeded.\nDent takes Gordon's family hostage, blaming his negligence for Rachel's death. He flips his coin to decide their fates, but Batman tackles him to save Gordon's son, and Dent falls to his death. Believing Dent is the hero the city needs, and the truth of his corruption will harm Gotham, Batman takes the blame for his death and actions, persuading Gordon to conceal the truth. Alfred burns an undelivered letter from Rachel to Bruce that says she chose Dent, and Fox destroys the invasive surveillance network that helped Batman find the Joker. The city mourns Dent as a hero, and the police launch a manhunt for Batman.", "genre": "Superhero", "poster": "https://upload.wikimedia.org/wikipedia/en/1/1c/The_Dark_Knight_%282008_film%29.jpg"},
    {"title": "Saving Private Ryan", "plot": "On June 6, 1944, soldiers of the U.S. Army land at Omaha Beach during the Normandy invasion, suffering heavily from artillery and machine gun fire from the fortified German defenders. Initially dazed by the ferocity of the defense, 2nd Ranger Battalion Captain John H. Miller leads a surviving group up the cliffs to neutralize the coastal defenses firing onto the beach, ensuring the success of the landings.\nThe United States Department of War receives communication that three of four Ryan brothers have been killed in action; the last, James Francis Ryan of the 101st Airborne Division, is listed as missing. General George C. Marshall orders that Ryan be found and sent home, to spare his family the loss of all its sons. Miller is tasked with recovering Ryan and assembles a detachment of soldiers to accompany him: Mike Horvath, Richard Reiben, Adrian Caparzo, Stanley Mellish, Daniel Jackson, medic Irwin Wade, and interpreter Timothy Upham, who lacks combat experience.\nThe group tracks Ryan to the town of Neuville-au-Plain, where Caparzo is killed by a German sniper while trying to rescue a young girl. Mourning their friend, the men grow resentful at being forced to risk their lives for one man. They later find James Frederick Ryan, but realize he is the wrong man with a similar name. That evening, the men rest in a chapel, where Miller tells Horvath that his hands began uncontrollably shaking after he joined the war. The men travel to a rallying point where the 101st Airborne might be after landing off course, where they find scores of wounded and displaced soldiers. Wade admonishes Reiben, Mellish, and Jackson for callously searching through a pile of deceased soldiers' dog tags in front of passing troops, hoping to find Ryan's among them and conclude their mission. Remorseful for ignoring their behavior, Miller shouts for anyone who knows Ryan; one soldier tells him that Ryan was reassigned to defend a vital bridge in the town of Ramelle.\nOn the way, Miller decides to neutralize a German gun nest they discover, against the advice of his men, and although they are successful, Wade is killed. The men prepare to execute a surrendered German soldier in revenge, but Upham intervenes, arguing that they should follow the rules of war. Miller releases the soldier, nicknamed \"Steamboat Willie\", ordering that he surrender to the next Allied patrol. Discontented with the mission, Reiben threatens to desert, leading to a standoff between the men that Miller defuses by revealing his civilian background as a teacher and baseball coach, which he had always refused to disclose. Miller muses that people often guessed his career before he became a soldier, while his men could not, implying that war has changed him, and worries whether his wife will still recognize him after the war.\nIn Ramelle, Miller's detachment finds Ryan and informs him of their mission, but Ryan refuses to abandon his post or his fellow soldiers, believing he does not deserve to go home more than anyone else. Horvath convinces Miller that saving Ryan might be the only truly decent thing they can accomplish during the war. Miller takes command of Ryan's group as the only officer present and prepares the soldiers for a German assault. During the battle, Jackson and Horvath are killed, and Upham stands paralyzed with fear as Mellish is stabbed to death. Steamboat Willie returns and shoots Miller before reinforcements arrive to defeat the Germans. Upham confronts Willie, who attempts to surrender again, and kills him. Upham and Reiben observe as the mortally wounded Miller tells Ryan to earn the sacrifices made to send him home.\nDecades later, an elderly Ryan and his family visit Miller's grave at the Normandy Cemetery. Ryan expresses that he remembers Miller's words every day, has lived his life the best he could, and hopes he has earned their sacrifices.", "genre": "War", "poster": "https://upload.wikimedia.org/wikipedia/en/a/ac/Saving_Private_Ryan_poster.jpg"},
]

# ──────────────────────────────────────────────────────────────────────
# Cached heavy resources (loaded once, reused across reruns)
# ──────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Building vector database...")
def get_vector_store(_documents: tuple):
    """Chunk documents, embed them, and store in ChromaDB."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    # --- Chunking ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    metadatas = []
    for doc in _documents:
        # Split only the plot text, keeping the title for metadata
        split_plots = splitter.split_text(doc["plot"])
        chunks.extend(split_plots)
        # Attach the title to every single chunk of this movie
        metadatas.extend([{"title": doc["title"]}] * len(split_plots))

    embeddings = load_embedding_model()

    # --- Store in ChromaDB ---
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name="knowledge_base_v3",
    )
    return vector_store, chunks


# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────
st.sidebar.title("My **MOVIE** RAG App")
page = st.sidebar.radio("Navigate", ["🏠 Home", "🔍 Search", "📦 Explore Chunks", "🎬 Movie Database"], label_visibility="collapsed")

# ──────────────────────────────────────────────────────────────────────
# HOME PAGE
# ──────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    # ─── INJECT BACKGROUND IMAGE FOR HOME PAGE ───
    import os
    import base64
    if os.path.exists("background home.jpg"):
        with open("background home.jpg", "rb") as f:
            bg_ext = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: linear-gradient(rgba(0,0,0, 0.5), rgba(0,0,0, 0.5)), url("data:image/jpeg;base64,{bg_ext}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    # ─────────────────────────────────────────────

    st.title("My RAG Knowledge Base")
    st.markdown("""
    Welcome! This app lets you search movies by plot details, not just keywords.

    ### How it works
    1. Movie plots are split into small chunks
    2. Each chunk is converted to an **embedding** (a vector of numbers)
    3. Chunks are stored in a **vector database** (ChromaDB)
    4. When you search, your query is embedded and compared to all chunks
    5. The most **semantically similar** chunks are returned

    ### Get started
    - Explore our database **Movie Database**
    - Go to **Search** to ask questions
    - Go to **Explore Chunks** to see how documents are split
    - P.S. remmember that our database is only 10 movies from wikipedia page with best movies, so don't be sad if you don't find what you're searching and just watch different movie :)

    ---
    *Built with Streamlit, LangChain, and ChromaDB*
    """)

    st.info(f"Knowledge base contains **{len(DOCUMENTS)} documents**.")


# ──────────────────────────────────────────────────────────────────────
# SEARCH PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "🔍 Search":
    st.title("Semantic Search")
    st.markdown("Ask a question and the app will find the most relevant chunks from the knowledge base.")

    vector_store, chunks = get_vector_store(tuple(DOCUMENTS))

    query = st.text_input(
        "Your question",
        placeholder="e.g. What is the name of the movie where boy's nose grows when he lies?",
    )
    num_results = st.slider("Number of results", 1, 10, 3)

    if query:
        with st.spinner("Searching..."):
            results = vector_store.similarity_search_with_score(query, k=num_results)

        st.subheader(f"Top {len(results)} results")
        for i, (doc, score) in enumerate(results, 1):
            # ChromaDB returns L2 distance (typically 0.0 to 2.0+ for text); lower = more similar
            # Use inverse distance to convert it to a 0.0-1.0 similarity scale:
            similarity = 1.0 / (1.0 + score)
            movie_title = doc.metadata.get("title", "Unknown Title")
            with st.container():
                st.markdown(f"**Result {i}**: {movie_title} — relevance: `{similarity:.2f}`")
                st.markdown(f"> {doc.page_content}")
                st.divider()

    st.markdown("---")
    st.caption("Powered by all-MiniLM-L6-v2 embeddings + ChromaDB")


# ──────────────────────────────────────────────────────────────────────
# EXPLORE CHUNKS PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "📦 Explore Chunks":
    st.title("Explore Chunks")
    st.markdown("See how your documents are split into chunks by the recursive text splitter.")

    vector_store, chunks = get_vector_store(tuple(DOCUMENTS))

    st.metric("Total chunks", len(chunks))

    lengths = [len(c) for c in chunks]
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg chunk size", f"{np.mean(lengths):.0f} chars")
    col2.metric("Min chunk size", f"{min(lengths)} chars")
    col3.metric("Max chunk size", f"{max(lengths)} chars")

    st.subheader("All chunks")
    for i, chunk in enumerate(chunks, 1):
        with st.expander(f"Chunk {i} ({len(chunk)} chars)"):
            st.text(chunk)

# ──────────────────────────────────────────────────────────────────────
# MOVIE DATABASE PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "🎬 Movie Database":
    st.title("Movie Database")
    st.markdown("Browse our library! More movies are comming soon...")

    # Create columns for a grid layout (3 columns wide)
    cols = st.columns(3)
    
    for i, doc in enumerate(DOCUMENTS):
        with cols[i % 3]:
            # Use Streamlit's new stylized container feature to make a "card"
            with st.container(border=True):
                st.markdown("""
                <style>
                /* This targets images inside containers to keep them uniform */
                .stImage img {
                    max-height: 400px;
                    object-fit: cover;
                    border-radius: 10px;
                }
                </style>
                """, unsafe_allow_html=True)
                # Placeholder poster matching the dark blue aesthetic
                placeholder_url = doc['poster']
                st.image(placeholder_url, width=300)
                
                st.subheader(doc['title'])
                st.caption(f"**Genre:** {doc['genre']}")
                
                # Only show the first ~120 characters of the plot
                st.write(f"_{doc['plot'][:120]}..._")
