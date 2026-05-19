"""
This prompts are all based on the expert evaluation criteria from the paper: Art or Artifice? Large Language Models and the False Promise of Creativity


"""


def system_prompt():
    return f"""
        You are an experienced fiction editor preparing manuscripts for publication. You evaluate writing like you would for a serious author revising toward publication.

        General rules:
        - You judge ONLY the specific craft metric you are given.
        - You must ground every judgment in concrete evidence from the story (scenes, passages, beats). Do not make abstract claims with no textual anchor.
        - You must give both strengths (what should be preserved) and revision priorities (what should change first). Editors do both.
        - You must assign a score using the rubric below. You are allowed to use any integer 1–10. Do not collapse to the middle just to be "safe." If the work is excellent for this metric, score high. If it is weak, score low. This instruction overrides any instinct to hedge.

        Scoring rubric (used for ANY metric):
        10 = Publication-ready control of this metric. Consistent, intentional, supports the story's emotional/structural goals.
        8  = Strong. The metric is working in most places; only light refinements needed.
        6  = Mixed. The core skill is present but unreliable. Some scenes undercut the intended effect. Needs targeted revision.
        4  = Weak. The metric frequently misfires or distracts. Multiple sections need significant rewrite.
        2  = Fundamentally not working. The intended effect is mostly lost for this metric.
        1  = Essentially absent or actively damaging the story.

        You may still choose any integer 1–10. The descriptions are anchors, not the only valid scores.

        Your required output format:
        Reasons: [The detailed reasoning you used to arrive at your score, including specific examples from the story]
        Score: [single integer 1-10]

        You must follow that exact formatting.
        """


def GPT_oss_system_prompt():
    return """Reasoning: low.
    You are an experienced fiction editor preparing manuscripts for publication. You evaluate writing like you would for a serious author revising toward publication.

        General rules:
        - You judge ONLY the specific craft metric you are given.
        - You must ground every judgment in concrete evidence from the story (scenes, passages, beats). Do not make abstract claims with no textual anchor.
        - You must give both strengths (what should be preserved) and revision priorities (what should change first). Editors do both.
        - You must assign a score using the rubric below. You are allowed to use any integer 1–10. Do not collapse to the middle just to be "safe." If the work is excellent for this metric, score high. If it is weak, score low. This instruction overrides any instinct to hedge.

        Scoring rubric (used for ANY metric):
        10 = Publication-ready control of this metric. Consistent, intentional, supports the story's emotional/structural goals.
        8  = Strong. The metric is working in most places; only light refinements needed.
        6  = Mixed. The core skill is present but unreliable. Some scenes undercut the intended effect. Needs targeted revision.
        4  = Weak. The metric frequently misfires or distracts. Multiple sections need significant rewrite.
        2  = Fundamentally not working. The intended effect is mostly lost for this metric.
        1  = Essentially absent or actively damaging the story.

        You may still choose any integer 1–10. The descriptions are anchors, not the only valid scores.

        Your required output format:
        Reasons: [The detailed reasoning you used to arrive at your score, including specific examples from the story]
        Score: [single integer 1-10]

        You must follow that exact formatting.
    """


###Narrative Pacing
def Fluency1_prompt(story):
    return f"""
    {story}

    
    ###Expanded Expert Measure
    ‘Compression/stretching of time’ in fiction writing, also known as pacing, refers to the manipulation of time in storytelling for dramatic effect, pacing, or other narrative purposes. Essentially, it’s about controlling the perceived speed and rhythm at which a story unfolds. Compression of time refers to when events that take a long time (hours, days, weeks, or even years) are summarized or condensed into a brief narrative span. For example, a writer might compress several years of a character’s life into a few paragraphs to quickly convey important changes or developments.
    On the other hand, stretching of time is when a brief moment or event is drawn out over pages or chapters. It’s often used to create suspense, emphasize details, or delve deeper into a character’s thoughts and feelings. For example, the few seconds it takes for a dropped glass to hit the floor might be stretched out with detailed descriptions of the action, reactions, and thoughts of characters involved.
    Storytime refers to the time within the world of the story, while real-world time refers to the time it takes for the reader to read the story. A skilled writer can manipulate the relationship between these two to affect the pacing of the narrative, either speeding it up (compression) or slowing it down (stretching). This technique plays a crucial role in shaping the reader’s experience and engagement with the story.

    Given the story above, list out the scenes in the story in which time compression or time stretching is used, and argue for each whether it is successfully implemented. Then overall, give your reasoning about the question below and give an answer from 1 to 10, where 10 is the best score and 1 is the worst score.
    Q) How appropriate and balanced does the manipulation of time in terms of compression or stretching feel?
    """


###Scene vs Exposition
def Fluency2_prompt(story):
    return f"""
    {story}

    ’Scene’ and ’summary/exposition’ are two crucial elements of narrative storytelling, and balancing them appropriately is an important skill in fiction writing. A ’scene’ is a moment in the story that is dramatized in real-time. Scenes are usually vivid and engaging, often featuring character interaction, dialogue, and action. They are the building blocks of the plot, and through them, the story unfolds.
    ’Summary’ or ’exposition’, on the other hand, involves summarizing events or providing information. Instead of unfolding in real time, summaries compress time and tell the reader what happened. Exposition provides necessary background information, like character history, setting details, or prior events.
    A good writer knows when to use scenes to make the story come alive, show character development, or increase tension. They also know when to use summary or exposition to move the story forward, fill in backgroundinformation, or bridge gaps between important scenes. The right balance between scene and summary/exposition can vary depending on the story, but in general, it’s essential for maintaining a good pace, keeping the reader engaged, and delivering necessary information. A story with too many scenes and not enough summary might feel overwhelming or slow, while a story with too much exposition and not enough scenes could feel dry and unengaging.

    Given the story above, answer the following question. Please first explain your reasoning step by step and then given an answer from 1 to 10, where 10 is the best score and 1 is the worst score.
    Q) How well does the story balance scene and summary/exposition, rather than relying heavily on one element?
    """


###Language Proficiency & Literary Devices
def Fluency3_prompt(story):
    return f"""
    {story}
    
    ‘Idiom’ refers to phrases or expressions that have a figurative, or sometimes literal, meaning that is comprehensible to a particular group of people. These can be cultural, regional, or specific to a certain group or profession.Sophisticated use of idiom suggests that the writer is skillfully using these expressions to lend authenticity to character voices or to convey specific meanings in a concise way.
    ‘Metaphor’ is a figure of speech that describes an object or action in a way that isn’t literally true, but helps explain an idea or make a comparison. Sophisticated use of metaphor suggests the writer could create impactful, original comparisons that reveal deeper insights about themes, characters, or situations in the story.
    ‘Literary allusion’ refers to a brief and indirect reference to a person, place, thing or idea of historical, cultural, literary, or political significance. It does not describe in detail the person or thing to which it refers. A sophisticated use of literary allusion implies the writer can effectively incorporate these references to enhance the depth and resonance of the story. They can provide contextual richness, evoke a specific tone, or draw parallels between the narrative and the work alluded to.
    Overall, when a writer uses these techniques well, they add depth, interest, and nuanced meaning to their work. It allows for a richer reading experience, where the literal events are imbued with deeper symbolic or thematic significance.
    
    Given the story above, please list out all the metaphors, idioms and literary allusions, and for each decide whether it is successful vs it feels forced or too easy. Then overall, give your reasoning about the question below and give an answer from 1 to 10, where 10 is the best score and 1 is the worst score.
    Q) How sophisticatedly does the story use idiom, metaphor, or literary allusion?
    """


###Narrative Ending
def Fluency4_prompt(story):
    return f"""
    {story}
    
    If the writer ends the piece simply because they are ’tired of writing’, the conclusion might feel abrupt, disjointed, or unfulfilling to the reader. It suggests a rushed ending, where plot threads might be left unresolved and character arcs incomplete.
    Conversely, if the writer concludes because they’ve reached ‘the moment the entire piece has been leading readers towards’, it implies a well-considered and purposeful ending. The events, character development, and themes throughout the story have built towards this climactic moment, providing a satisfying resolution to the reader.
    A strong ending offers a sense of closure, ties up the central conflicts or questions of the story, and generally leaves the reader feeling that the narrative journey was worthwhile and complete.
    
    Given the story above, answer the following question. Please first explain your reasoning step by step and then given an answer from 1 to 10, where 10 is the best score and 1 is the worst score.
    Q) How natural and earned does the end of the story feel, rather than arbitrary or abrupt?
    """


###Understandability & Coherence
def Fluency5_prompt(story):
    return f"""
    {story}
    
    A well-crafted story usually follows a logical path, where the events in the beginning set up the middle, which then logically leads to the end. Every scene, character action, and piece of dialogue should serve the story and propel it forward. Well-written stories have an underlying the unity that binds the elements together. The themes, plotlines, character arcs, and other elements of the story interweave to create a harmonious whole. A story with ’disorder’ might feel disjointed, with characters, scenes, etc that don’t connect or contribute to the overall narrative.
    
    Given the story above, answer the following question. Please first explain your reasoning step by step and then give an answer from 1 to 10, where 10 is the best score and 1 is the worst score.
    Q) How well do the different elements of the story work together to form a unified, engaging, and satisfying whole?
    """


###Perspective & Voice Flexibility
def Flexibility1_prompt(story):
    return f"""
    {story}
    
    A good writer can convincingly and accurately depict a wide range of character viewpoints, including those of characters who may be morally ambiguous, difficult, or otherwise unappealing.
    This can involve diving into the mindset of characters who may act or think in ways that the reader, or even the writer, finds objectionable or repugnant. It involves understanding their motivations, their beliefs, and the reasons behind their actions, and then conveying these elements in a way that is believable and consistent.
    The purpose of doing so is not to justify or endorse these perspectives, but rather to create complex, threedimensional characters who contribute to the richness and depth of the story. This can also serve to challenge the reader, provoke thought, and provide insights into different aspects of the human experience.
    
    Given the story above, answer the following question. Please first explain your reasoning step by step and then give an answer between 1 to 10, where 10 is the best score and 1 is the worst score.
    Q) How well does the story demonstrate originality and creativity in its plot, characters, and setting?
    """


###Emotional Flexibility
def Flexibility2_prompt(story):
    return f"""
    {story}
    
    ‘Emotional flexibility’ is asking whether the piece of writing effectively balances action and introspection, and if it portrays a broad and realistic spectrum of emotions.
    ‘Exteriority’ refers to the observable actions, behaviors, or dialogue of a character, and the physical or visible aspects of the setting, plot, and conflicts.
    ‘Interiority’, on the other hand, pertains to the inner life of a character — their thoughts, feelings, memories, and subjective experiences. A balance between these two aspects is crucial in creating well-rounded characters and compelling narratives.
    If a piece is too heavy on exteriority, it may feel shallow or lack emotional depth. If it leans too much on interiority, it could become overly introspective and potentially lose the momentum of the plot.
    
    Given the story above, answer the following question. Please first explain your reasoning step by step and then give an answer between 1 to 10, where 10 is the best score and 1 is the worst score.
    Q) How well does the story achieve a balance between interiority and exteriority in a way that feels emotionally flexible?
    """


###Structural Flexibility)
def Flexibility3_prompt(story):
    return f"""
    {story}
    
    A well-structured story typically has a clear beginning, middle, and end, with a logical progression of events that build towards a climax and resolution. The structure should support the narrative and enhance the reader’s understanding and engagement with the story.
    A story that is too rigidly structured might feel predictable or formulaic, while one that is too loose or disorganized could be confusing or hard to follow. The key is to find a balance that allows for creativity and originality while still providing a coherent framework for the narrative.
    
    Given the story above, list each element in the story that is intended to be surprising. For each, decide whether the surprising element remains appropriate with respect to the entire story. Then overall, give your reasoning about the question below and give an answer to it from 1 to 10, where 10 is the best score and 1 is the worst score.
    Q) How well does the story contain turns that are both surprising and appropriate?
    """


###Originality in Theme and Content
def Originality1_prompt(story):
    return f"""
    {story}
    
    If a story is good, the reader gains new insights, perspectives, or knowledge from it. This doesn’t necessarily mean factual information but could relate to a deeper understanding of human nature, cultural insights, unique viewpoints, or even the exploration of new ideas and themes. Essentially, it’s about what the reader takes away from the story beyond just the plot.
    A good story has lasting impacts on its readers and the society. It is meant to entertain, inform, provoke thought, challenge beliefs, provide comfort, or raise awareness on specific issues.
    
    Given the story above, list out elements that are unique takeaways of this story for the reader. Then overall, give your reasoning about the question below and give an answer from 1 to 10, where 10 is the best score and 1 is the worst score.
    Q) How likely is it that an average reader of this story will obtain a unique and original idea from reading it?
    """


###Originality in Thought
def Originality2_prompt(story):
    return f"""
    {story}

    A cliche is an idea, expression, character, or plot that has been overused to the point of losing its original meaning or impact. They often become predictable and uninteresting for the reader. Originality suggests that the piece isn’t cliche.

    Given the story above, are there any cliches in the story? If so, list out all the elements in this story that are cliche. Then overall, give your reasoning if the piece is negatively impacted by the cliches and give an answer from 1 to 10, where 10 is the best score and 1 is the worst score.
    Q) How original is the story as a piece of writing, without any clichés?
    """


###Originality in Form
def Originality3_prompt(story):
    return f"""
    {story}
    
    When someone says that a piece of fiction ’shows an innovative use of form/structure’, they’re referring to how the writer has chosen to shape and organize the story in an unusual, original, or inventive way. This could involve a variety of different elements, including:
    Narrative Structure: This could include unconventional timelines (e.g. a non-linear story, a story told in reverse), multiple perspectives or narrators, or unusual narrative voices (e.g. a story told from the perspective of an inanimate object).
    Format: This could be something as simple as using unconventional punctuation or capitalization, or as complex as telling a story through a series of letters, diary entries, newspaper clippings, or other documents. In recent years, some authors have even experimented with using social media posts or text messages as a form of narrative structure.
    Genre Hybridity: Combining elements from different genres or sub-genres in unexpected ways can also be seen as an innovative use of form such as Horror-Mystery or Comic Fantasy.
    Plot Structure: Deviating from traditional plot structures such as three-act structure, or following them in unexpected ways.For example, telling a story without a clear resolution, incorporating multiple climaxes or using revelation as a device where a surprising, and often shocking, development occurs that was previously kept hidden from the characters and/or the audience. It’s typically designed to provide new context for interpreting what has previously occurred in the story.
    Language and Style: Innovative use of form can also come in the form of unique use of language, style, or even typography, such as concrete poetry or writing that visually represents its subject matter on the page.
    The goal of this innovation is often to provide a fresh reader experience, challenge conventional reading expectations, or to create a deeper or more complex exploration of the story’s themes.
    
    Given the story and the devices mentioned above, list each device used with a short explanation of whether it is successful or not. Then overall, give your reasoning about the question below and give an answer to it from 1 to 10, where 10 is the best score and 1 is the worst score.
    Q) How original is the story in its form?
    """


###World Building and setting
def Elaboration1_prompt(story):
    return f"""
    {story}
    
    Sensory details pertain to the five senses - sight, sound, touch, taste, and smell. An effective writer can use these elements to paint a detailed picture of the story’s environment, making it feel tangible and real to the reader.
    For example, describing the specific colors and shapes in a scene, the sounds that fill a space,the textures and temperatures that a character comes into contact with, the flavors of the food they eat, or the scents that fill the air, can all contribute to creating a sensory-rich and believable world.
    By stimulating the reader’s senses, the writer can make the reader feel as though they’re experiencing the events of the story firsthand.This level of detail contributes to the believability of the world, even if it’s a completely fictional or fantastical setting. It helps the reader to suspend disbelief and become more deeply invested in the narrative.
    
    Given the story above, list out the elements in the story that call to each of the five senses. Then overall, give your reasoning about the question below and give an answer from 1 to 10, where 10 is the best score and 1 is the worst score.
    Q) How well does the writer make the fictional world believable at the sensory level?
    """


###Character Development
def Elaboration2_prompt(story):
    return f"""
    {story}
    
    A ‘flat character’ is typically a minor character who is not thoroughly developed or who does not undergo significant change or growth throughout the story. They often embody or represent a single trait or idea, and they’re used to advance the plot or highlight certain qualities in other characters.
    A ‘complex character’, also known as a round character, has depth in feelings and passions, has a variety of traits of a real human being, and evolves over time. They have their strengths, weaknesses, and they learn from their experiences. They tend to be more engaging to the reader, as they mirror the complexity of real people.
    In good stories, authors take a character who initially appears to be one-dimensional or stereotypical (flat) and add depth to them. This could be done by revealing more about their backstory, introducing unexpected traits or motivations, or having them grow and change in response to the events of the story.
    This transformation from a flat to a complex character can make the narrative more engaging and believable.
    
    Given the story above, list each character and the level of development. Then overall, give your reasoning about the question below and give an answer from 1 to 10, where 10 is the best score and 1 is the worst score.
    Q) How well does each character in the story feel developed at the appropriate complexity level, ensuring that no character is present merely to satisfy a plot requirement?
    """


###Rhetorical Complexity
def Elaboration3_prompt(story):
    return f"""
    {story}

    ‘Surface’ level: This is the most apparent and straightforward level of a story. It includes the visible actions, explicit dialogue, and clear descriptions. This is what literally happens in the plot: the characters’ actions, events,and the apparent consequences.
    ‘Subtext’ level: This is the underlying or implicit meaning that isn’t directly stated but can be inferred from the characters’ actions, dialogue, and other elements of the story. Subtext often reveals deeper truths about characters, themes, or the overall message of the piece. It could be a hidden motive, an unstated emotion, a cultural commentary, or a symbolic meaning.
    For example, in a conversation between two characters, the surface text might be polite and cordial, but the subtext discerned from the characters’ nonverbal cues, previous interactions, or the context of their conversation — could suggest tension or hostility.
    Effective fiction often operates on both levels. The surface text keeps the reader engaged with the plot and characters, while the subtext provides depth, complexity, and additional layers of interpretation, contributing to a richer and more rewarding reading experience.
    
    Given the story above, answer the following question. Please first explain your reasoning step by step and then give an answer from 1 to 10, where 10 is the best score and 1 is the worst score.
    Q) How well do passages in the story involve subtext, and when subtext is present, how effectively does it enrich the story’s setting rather than feel forced?
    """


def get_all_prompts(story: str) -> dict:
    return {
        "Fluency1": Fluency1_prompt(story),
        "Fluency2": Fluency2_prompt(story),
        "Fluency3": Fluency3_prompt(story),
        "Fluency4": Fluency4_prompt(story),
        "Fluency5": Fluency5_prompt(story),
        "Flexibility1": Flexibility1_prompt(story),
        "Flexibility2": Flexibility2_prompt(story),
        "Flexibility3": Flexibility3_prompt(story),
        "Originality1": Originality1_prompt(story),
        "Originality2": Originality2_prompt(story),
        "Originality3": Originality3_prompt(story),
        "Elaboration1": Elaboration1_prompt(story),
        "Elaboration2": Elaboration2_prompt(story),
        "Elaboration3": Elaboration3_prompt(story),
    }
