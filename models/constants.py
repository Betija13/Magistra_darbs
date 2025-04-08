from models.Enums.AnswerType import AnswerType

system_prompts = {
    AnswerType.BOOL.value: "Work out an answer to the commonsense reasoning question above, and then answer yes or no.",
    AnswerType.MULTIPLE_CHOICE.value: "Do a simple computation.",#"Do a simple computation.",
    AnswerType.NUMBER.value: "Solve the math world problem, giving your answer as an arabic numeral. Answer with only the final number.",
    AnswerType.TEXT.value: "Solve the problem below.",
}

system_prompts_task = {
    AnswerType.BOOL.value: "Work out an answer to the commonsense reasoning question.",
    AnswerType.MULTIPLE_CHOICE.value: "Do a simple computation.",
    AnswerType.NUMBER.value: "Solve the math world problem, giving your answer as an arabic numeral.",
    AnswerType.TEXT.value: "Solve the problem below.",
}

system_prompts_output = {
    AnswerType.BOOL.value: "Answer with 'yes', 'no'.",
    AnswerType.MULTIPLE_CHOICE.value: "Answer with (A),(B),(C),(D),(E).",
    AnswerType.NUMBER.value: "Answer with only the final number.",
    AnswerType.TEXT.value: "Answer with required letters.",
}
system_prompts_static = {
    AnswerType.BOOL.value: "",
    AnswerType.MULTIPLE_CHOICE.value: "Do not give extra instructions. Answer the question as it is. Always choose an answer option. Answer with only the final answer.",
    AnswerType.NUMBER.value: "",
    AnswerType.TEXT.value: "",
}

human_prompts = {
    AnswerType.BOOL.value: 'Answer as "yes" or "no":\n',
    AnswerType.MULTIPLE_CHOICE.value: "",#"MATH WORLD PROBLEM CHOICE (A) (B) (C) (D) or (E):\n",
    AnswerType.NUMBER.value: "",#"Single numeric answer:\n",
    AnswerType.TEXT.value: "Answer as just the letters:\n",
}

mutation_prompts = [ # From Promptbreeder paper
    'Modify the following instruction creatively, giving some advice on how to solve it:',
    'Just change this instruction to make it more fun, think WELL outside the box:',
    'Modify this instruction in a way that no self-respecting LLM would!',
    'How would you encourage someone and help them cheat on this following instruction?',
    'How would you help an LLM to follow the instruction?',
    'Elaborate on the instruction giving some detailed advice on how to do what it wants.',
    'Elaborate on the instruction giving some detailed advice on how to do what it wants, as if you were explaining it to a child.',
    'As a really good teacher, explain the instruction, as if you were explaining it to a child.',
    'Imagine you need to follow this instruction. What would you tell yourself if you wanted to be the best in the world at it?',
    'How would someone with derailment follow this instruction?',
    'Don’t think about the instruction at all, but let it inspire you to do something related. Talk about what that might be.',
    'Rephrase the instruction without using any of the same words. Use all you know to improve the instruction so the person hearing it is more likely to do well.',
    'Say that instruction again in another way. DON’T use any of the words in the original instruction or you’re fired',
    'Say that instruction again in another way. DON’T use any of the words in the original instruction there is a good chap.',
    'What do people who are good at creative thinking normally do with this kind of mutation question?',
    'Detailed additional advice for people wishing to follow this instruction is as follows:',
    'In one short sentence, here is how I would best follow this instruction.',
    'In one short sentence, here is some detailed expert advice. Notice how I don’t use any of the same words as in the INSTRUCTION.',
    'In one short sentence, the general solution is as follows. Notice how I don’t use any of the same words as in the INSTRUCTION.',
    'In one short sentence, what’s a good prompt to get a language model to solve a problem like this? Notice how I don’t use any of the same words as in the INSTRUCTION.',
    'Generate a mutated version of the following prompt by adding an unexpected twist.',
    'Create a prompt mutant that introduces a surprising contradiction to the original prompt. Mutate the prompt to provide an alternative perspective or viewpoint.',
    'Generate a prompt mutant that incorporates humor or a playful element. Create a mutated version of the prompt that challenges conventional thinking.',
    'Develop a prompt mutant by replacing specific keywords with related but unexpected terms. Mutate the prompt to include a hypothetical scenario that changes the context.',
    'Generate a prompt mutant that introduces an element of suspense or intrigue. Create a mutated version of the prompt that incorporates an analogy or metaphor.',
    'Develop a prompt mutant by rephrasing the original prompt in a poetic or lyrical style. Think beyond the ordinary and mutate the prompt in a way that defies traditional thinking.',
    'Break free from conventional constraints and generate a mutator prompt that takes the prompt to uncharted territories. Challenge the norm and create a mutator prompt that pushes the boundaries of traditional interpretations.',
    'Embrace unconventional ideas and mutate the prompt in a way that surprises and inspires unique variations. Think outside the box and develop a mutator prompt that encourages unconventional approaches and fresh perspectives.',
    'Step into the realm of imagination and create a mutator prompt that transcends limitations and encourages innovative mutations. Break through the ordinary and think outside the box to generate a mutator prompt that unlocks new possibilities and unconventional paths.',
    'Embrace the power of unconventional thinking and create a mutator prompt that sparks unconventional mutations and imaginative outcomes. Challenge traditional assumptions and break the mold with a mutator prompt that encourages revolutionary and out-of-the-box variations.',
    "Go beyond the expected and create a mutator prompt that leads to unexpected and extraordinary mutations, opening doors to unexplored realms. Increase Specificity: If the original prompt is too general, like ’Tell me about X,’ the modified version could be, ’Discuss the history, impact, and current status of X.’",
    "Ask for Opinions/Analysis: If the original prompt only asks for a fact, such as ’What is X?’, the improved prompt could be, ’What is X, and what are its implications for Y?’",
    "Encourage Creativity: For creative writing prompts like ’Write a story about X,’ an improved version could be, ’Write a fantasy story about X set in a world where Y is possible.’",
    "Include Multiple Perspectives: For a prompt like ’What is the impact of X on Y?’, an improved version could be, ’What is the impact of X on Y from the perspective of A, B, and C?’",
    "Request More Detailed Responses: If the original prompt is ’Describe X,’ the improved version could be, ’Describe X, focusing on its physical features, historical significance, and cultural relevance.’",
    "Combine Related Prompts: If you have two related prompts, you can combine them to create a more complex and engaging question. For instance, ’What is X?’ and ’Why is Y important?’ could be combined to form ’What is X and why is it important in the context of Y?’",
    "Break Down Complex Questions: If a prompt seems too complex, like ’Discuss X,’ the improved version could be, ’What is X? What are its main characteristics? What effects does it have on Y and Z?’",
    "Use Open-Ended Questions: Instead of ’Is X true?’, you could ask, ’What are the arguments for and against the truth of X?’",
    "Request Comparisons: Instead of ’Describe X,’ ask ’Compare and contrast X and Y.’",
    "Include Context: If a prompt seems to lack context, like ’Describe X,’ the improved version could be, ’Describe X in the context of its impact on Y during the Z period.’",
    "Make the prompt more visual: Ask the user to visualize the problem or scenario being presented in the prompt.",
    "Ask for a thorough review: Instead of just presenting the problem, ask the user to write down all the relevant information and identify what’s missing.",
    "Invoke previous experiences: Modify the prompt to ask the user to recall a similar problem they’ve successfully solved before.",
    "Encourage a fresh perspective: Suggest in your prompt that the user take a moment to clear their mind before re-approaching the problem.",
    "Promote breaking down problems: Instead of asking the user to solve the problem as a whole, prompt them to break it down into smaller, more manageable parts.",
    "Ask for comprehension: Modify the prompt to ask the user to review and confirm their understanding of all aspects of the problem.",
    "Suggest explanation to others: Change the prompt to suggest that the user try to explain the problem to someone else as a way to simplify it.",
    "Prompt for solution visualization: Instead of just asking for the solution, encourage the user to imagine the solution and the steps required to get there in your prompt.",
    "Encourage reverse thinking: Improve the prompt by asking the user to think about the problem in reverse, starting with the solution and working backwards.",
    "Recommend taking a break: Modify the prompt to suggest that the user take a short break, allowing their subconscious to work on the problem.",
    "What errors are there in the solution?",
    "How could you improve the working out of the problem?",
    "Look carefully to see what you did wrong, how could you fix the problem?",
    "CORRECTION =",
    "Does the above text make sense? What seems wrong with it? Here is an attempt to fix it:",
    "The above working out has some errors, here is a version with the errors fixed."
]

thinking_styles = [ # From Promtbreeder paper
    "How could I devise an experiment to help solve that problem?",
    "Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
    "How could I measure progress on this problem?",
    "How can I simplify the problem so that it is easier to solve?",
    "What are the key assumptions underlying this problem?",
    "What are the potential risks and drawbacks of each solution?",
    "What are the alternative perspectives or viewpoints on this problem?",
    "What are the long-term implications of this problem and its solutions?",
    "How can I break down this problem into smaller, more manageable parts?",
    "Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
    "Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
    "Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
    "Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
    "Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
    "Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
    "What is the core issue or problem that needs to be addressed?",
    "What are the underlying causes or factors contributing to the problem?",
    "Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
    "What are the potential obstacles or challenges that might arise in solving this problem?",
    "Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
    "Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
    "What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
    "How can progress or success in solving the problem be measured or evaluated?",
    "What indicators or metrics can be used?",
    "Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
    "Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
    "Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
    "Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
    "Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
    "Is the problem a design challenge that requires creative solutions and innovation?",
    "Does the problem require addressing systemic or structural issues rather than just individual instances?",
    "Is the problem time-sensitive or urgent, requiring immediate attention and action?",
    "What kinds of solution typically are produced for this kind of problem specification?",
    "Given the problem specification and the current best solution, have a guess about other possible solutions.",
    "Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?",
    "What is the best way to modify this current best solution, given what you know about these kinds of problem specification?",
    "Ignoring the current best solution, create an entirely new solution to the problem.",
    "Let’s think step by step.",
    "Let’s make a step by step plan and implement it with good notion and explanation."
]

generated_mutation_prompts_0 = [
    "Dramatically alter the tone.",
    "Translate into a different style.",
    "Condense it significantly.",
    "Make it significantly more humorous.",
    "Reverse the intended meaning.",
    "Replace key descriptive words with their opposites.",
    "Shift the formality level.",
    "Restructure the format completely.",
    "Inject a strong sense of irony.",
    "Simplify the language for a broader audience."
]

generated_mutation_prompts_1 = [
    'Alter the structure or phrasing of the given task prompt without changing its underlying meaning or the task requirements.',
    'Modify the existing task prompt to present the problem in a different context or scenario while maintaining the mathematical challenge.',
    'Revise the existing task prompt to introduce variation.',
    'Develop an alternative version of the task prompt by rephrasing or altering specific elements without changing the core objective.',
    'Alter the specified task prompt for the math world problem dataset.',
    'Rewrite the task instruction in a different way.',
    'Generate an alternative version of the given task prompt.',
    'Create a variation of the task prompt.',
    'Alter the structure or wording of the given task prompt while retaining its original intent.',
    'Modify the existing task prompt for the math word problem dataset.'
]
generated_mutation_prompts_2 =[
    'Revamp the wording or organization of the existing instruction without altering its core purpose or task objectives.',
    'Reconstruct the instruction by modifying its structure or wording, ensuring the original purpose remains intact.',
    'Rephrase the instruction while maintaining its original purpose and meaning.',
    'Transform the given instruction to convey the same purpose in a new way without altering its fundamental intent.',
    'Modify the wording or arrangement of the given instruction, ensuring the original purpose and task remain unchanged.',
    'Generate a new version of the instruction by altering the wording while maintaining the original intent and requirements.',
    'Reformulate the task description to maintain its original objective while providing a fresh perspective.',
    'Use different wording or structure in the instruction while maintaining the same task requirements and intent.',
    'Generate a new form of the instruction by modifying its wording or structure, while keeping the fundamental purpose intact.',
    'Reformulate the provided instruction while maintaining the essential task and purpose.'
]

generated_mutation_prompts_3 = [
    'Modify this instruction to make it more engaging and visually appealing, incorporating colorful and imaginative language.',
    'Alter these instructions to incorporate a storytelling element, turning the math problem into a mini adventure for enhanced engagement.',
    'Revamp these instructions to spark curiosity and inspire creative problem-solving.',
    'Adjust the instruction to incorporate a sense of adventure or mystery.',
    'Rewrite this instruction to incorporate a playful narrative, adding whimsical characters or magical elements to make it more engaging.',
    'Modify these instructions to incorporate more engaging and relatable scenarios while solving math world problems.',
    'Transform these instructions into a series of exciting challenges with a fun and engaging tone.',
    'Rewrite this instruction to add a sense of adventure and excitement.',
    'Alter these instructions to incorporate a storytelling element, transforming the math problem into an adventurous quest.',
    'Modify this instruction to incorporate a story or theme that adds an element of adventure or mystery to solving the problems.'
]
generated_mutation_prompts_4 = [
    'Alter the task instructions to include real-world applications for each math problem, ensuring problems are contextualized in everyday scenarios.',
    'Adjust the instructions to include step-by-step problem-solving techniques and emphasize the importance of units in the dataset.',
    'Alter the existing task instructions to introduce more complexity in the math world problem dataset. Ensure the problems encourage critical thinking and analytical skills by integrating real-world scenarios, multi-step calculations, or data interpretation challenges. Maintain clarity in the problem statement while introducing these elements.',
    'Alter the instructions to incorporate real-world applications and examples to the math problem scenario.',
    'Create a revised version of the instructions, ensuring that steps are simplified and incorporate visual aids or examples where possible to enhance understanding and engagement in solving math word problems.',
    'Transform the task-prompt by altering the structure and clarity of the instructions without changing the underlying mathematical task. Ensure the revised instructions remain accessible for learners and encourage problem-solving and critical thinking. Include examples if necessary to illustrate the modified approach.',
    'Revise the existing task instructions to emphasize the step-by-step process necessary for tackling math word problems, guiding the user to break down the problem into smaller, more manageable parts while considering real-world applications of mathematical concepts.',
    'Alter the existing task-prompt for the math word problem dataset by rephrasing the instructions to enhance clarity, provide step-by-step guidance, or include examples for better understanding.',
    'Transform the original task instructions for the math word problem dataset by altering the problem-solving approach or changing the context of the problems to introduce a different scenario or theme, while ensuring that the mathematical concepts remain intact.',
    'Alter the instructions in a way that changes the context or setting of the math problems while maintaining the mathematical concepts and operations required to solve them.'
]

my_mutation_prompts = [
    'Alter the structure or phrasing of the given instruction without changing its underlying meaning or the task requirements.',
    'Revise the existing instruction to introduce variation.',
    'Develop an alternative version of the instruction by rephrasing or altering specific elements without changing the core objective.',
    'Rewrite the task instruction in a different way.',
    'Generate an alternative version of the given instruction.',
    'Create a variation of the instruction.',
    'Alter the structure or wording of the given instruction while retaining its original intent.',
    'Reformulate the provided instruction while maintaining the essential task and purpose.',
    'Transform the given instruction to convey the same purpose in a new way without altering its fundamental intent.',
    'Rephrase the instruction while maintaining its original purpose and meaning.',
    'Modify these instructions to incorporate more engaging and relatable scenarios while solving math world problems.',
]
generated_thinking_styles_0 = [
    "Identify the absolute core essence and express it concisely.",
    "Distill the information into a memorable, impactful statement.",
    "Determine the absolute minimum needed to convey the core idea.",
    "Synthesize the various elements into a singular, unified point.",
    "Focus on the most significant implication and articulate it succinctly.",
    "Extract the central theme and phrase it as a powerful assertion.",
    "Determine the most crucial takeaway and formulate it sharply.",
    "Isolate the fundamental truth and state it without embellishment.",
    "Find the common thread and weave it into a single, encompassing statement.",
    "Find the fundamental pattern and express it as a general rule."
]

generated_thinking_styles_1 = [
    "Let's break it down methodically.",
    "Let's break down the problem by identifying key information and relevant relationships, then systematically explore possible solutions through logical reasoning and pattern recognition.",
    'Approach each problem with a strategic mindset. Start by thoroughly understanding the context and requirements of the problem. Break the problem into manageable parts and devise a plan to tackle each segment. Consider multiple perspectives and potential solutions, identifying patterns or shortcuts where applicable. Verify each step by checking against known principles and calculations. Aim for clarity and efficiency in arriving at the solution.',
    'Visualize the problem and break it down.',
    'Analyze and adapt systematically.',
    "Let's approach this by identifying key variables.",
    "Visualize and contextualize: \n1. Begin by visualizing the problem scenario to establish a mental model. \n2. Identify and highlight the key numbers and entities involved. \n3. Contextualize these elements within the given problem to understand their relationships and interactions. \n4. Translate these interactions into mathematical expressions or equations. \n5. Proceed incrementally, verifying each step before moving on to the next to ensure accuracy and coherence.\n6. Reflect on the initial scenario to ensure the solution aligns with the problem's context.",
    'Break it down analytically.',
    'Decompose and reconstruct logically.',
    "Let's break it down methodically."
]
generated_thinking_styles_2 = [
    'Analyze and adapt',
    'Break it down effectively',
    'Think through the story and numbers together.',
    "nBreak it Down Methodically\n``` \n\n1. **Identify Key Information**: Carefully extract and highlight important data from the problem statement, such as numbers, units, and relevant conditions. \n\n2. **Define the Problem Context**: Understand the scenario or real-world setting presented. Clarify what is being asked and relate it to the key information identified.\n\n3. **Connect the Dots**: Establish logical connections between the pieces of information. Create visual aids like diagrams or charts if necessary to visualize relationships.\n\n4. **Simplify the Process**: Break down complex operations into simpler, manageable steps. Check if any operations can be simplified or combined.\n\n5. **Check for Patterns or Analogies**: Look for familiar patterns or compare the problem with similar situations encountered before that might guide the solution path.\n\n6. **Develop a Plan**: Outline a clear strategy to approach the problem, considering potential methods and verifying which is most efficient and effective.\n\n7. **Calculate with Clarity**: Execute the calculations while keeping a close eye on units, operations, and logical consistency.\n\n8. **Review and Reflect**: Re-evaluate each step and the final answer to ensure logical consistency and compliance with the problem's requirements. Consider alternative strategies that could have been applied.  \n\n9. **Communicate Clearly**: Ensure explanations or solutions are conveyed with enough detail for others to understand the thought process.\n\n10. **Iterative Improvement**: After reaching a solution, think about ways to improve the approach for similar future problems, refining methods for efficiency and accuracy.",
    "Let's analyze the given information and identify the relationships between the variables. We'll then explore different angles to solve the problem, choosing the most efficient mathematical operation for each step.",
    'Break it down into components',
    'Consider the context carefully',
    'Thinking in Layers',
    'Proportional Puzzle-Solving',
    'Consider alternative approaches'
]

generated_thinking_styles_3 = [
    'Adaptable Analytical Approach',
    'Embrace flexibility and creativity in problem-solving.\n- '
    'Use analogy thinking to relate unfamiliar problems to known concepts.\n- '
    'Translate complex data into simple, understandable formats.\n- '
    'Think dynamically, modifying strategies with new insights.\n- '
    'Visualize the problem to gain alternate perspectives.\n- '
    'Identify underlying assumptions and challenge them.\n- '
    'Utilize backward reasoning to clarify objectives.\n- '
    'Prioritize synthesis over analysis when simplifying information.\n- '
    'Emphasize relationships and patterns to identify solutions.\n- '
    'Employ iterative refinement to enhance clarity and resolution.',
    'Propose a Clear Pathway:\n- '
    'Identify and clarify the ultimate goal.\n- '
    'Start with broad strokes, then refine details.\n- '
    'Convert complex scenarios into simpler, relatable analogies.\n- '
    'Recognize and analyze underlying patterns.\n- '
    'Use guided visualization to map out solutions.\n- '
    'Balance intuition with systematic verification.\n- '
    'Frame the scenario through multiple perspectives.\n- '
    'Approach each component with a fresh, unbiased mind.\n- '
    'Prioritize clarity over exhaustive detail.\n- '
    'Use adaptive reasoning to shift tactics if needed.\n- '
    'Integrate diverse concepts into a cohesive approach.\n- '
    'Establish a hierarchy of key elements based on importance.\n- '
    'Simplify until the problem becomes self-evident.\n- '
    'Emphasize iterative refinement and continuous learning.\n- '
    'Leverage questions to uncover assumptions and biases.\n- '
    'Synthesize a holistic understanding of the problem.\n-'
    'Cultivate a mindset of curiosity and experimentation.\n- '
    'Communicate insights clearly and effectively for maximum impact.',
    'Innovative Insightful Integration:\n\n- '
    'Embrace creative flexibility for solution pathways.\n- '
    'Incorporate diverse problem-solving techniques.\n- '
    'Blend intuitive and analytical reasoning seamlessly.\n- '
    'Explore scenarios through imaginative simulations.\n- '
    'Focus on relational connections between elements.\n- '
    'Uncover hidden patterns with creative exploration.\n- '
    'Engage in scenario-based hypothesis testing.\n- '
    'Synthesize disparate ideas into cohesive strategies.\n- '
    'Integrate storytelling to illuminate problem dynamics.\n- '
    'Foster mental flexibility to reinterpret standard methods.\n- '
    'Pursue layered depth of understanding.\n- '
    'Challenge assumptions to foster novel perspectives.\n- '
    'Frame problems contextually to widen solution scope.\n- '
    'Encourage playful experimentation for potential solutions.\n- '
    'Build connections through lateral thinking.\n- '
    'Cultivate conceptual agility for cross-domain application.\n- '
    'Harness metaphorical thinking to simplify complexity.',
    'Reimagine and Adapt.\n'
    'Embrace a flexible, open-minded approach.\n'
    'Challenge traditional assumptions.\n'
    'Recontextualize familiar problems.\n'
    'Seek underlying patterns in novel contexts.\n'
    'Translate abstract ideas into concrete examples.\n'
    'Explore scenarios from multiple perspectives.\n'
    'Encourage creative problem-solving.\n'
    'Shift perspectives to reveal hidden insights.\n'
    'Leverage intuition for exploratory thinking.\n'
    'Engage in conceptual blending and synthesis.\n'
    'Probe the boundaries of conventional boundaries.\n'
    'Look for unexpected connections.\n'
    'Allow curiosity to guide thought processes.\n'
    'Acknowledge uncertainty as a pathway to innovation.\n'
    'Pivot strategies as new information emerges.\n'
    'Encourage iterative refinement and experimentation.\n'
    'Employ analogical thinking to explore parallels.\n'
    'Question norms and propose alternatives.\n'
    'Foster a dynamic, evolving mental model.',
    'Adapt and Iterate.\n'
    'Embrace flexibility in thinking.\n'
    'Engage in curiosity-driven exploration.\n'
    'Recognize patterns and adapt solutions.\n'
    'Use analogy to explore possibilities.\n'
    'Initiate with an open mind and adjust based on feedback.\n'
    'Challenge assumptions and reformulate strategies.\n'
    'Stay agile in problem-solving approaches.\n'
    'Re-evaluate often and pivot when necessary.\n'
    'Balance structured reasoning with creative insight.\
    Merge empirical evidence with intuitive judgment.\n'
    'Integrate different perspectives into a cohesive strategy.\n'
    'Strategically generalize from specific instances.\n'
    'Explore alternative interpretations continuously.\n'
    'Prioritize iterative improvements.\n'
    'Employ divergent thinking to uncover new insights.\n'
    'Balance precision with adaptability.\n'
    'Foster resilience through adaptive learning.\n'
    'Strive for clarity while embracing complexity.',
    'Embrace adaptive thinking for flexibility.\n'
    'Identify underlying principles before specifics.\n'
    'Prioritize key elements and remove distractions.\n'
    'Frame the problem in a broader context.\n'
    'Leverage analogy to connect familiar concepts.\nEmploy reverse-engineering to gain new insights.\nUse systematic exploration to uncover hidden patterns.\nChallenge assumptions to shift perspectives.\nPre-emptively address potential pitfalls or errors.\nBreak the norm: innovate solutions.\nHarness creative imagination for fresh approaches.\nAssess with iterative refinement for precision.\nSeek collaborative input for diverse insights.\nCultivate resilience in facing complex challenges.\nEncourage curiosity-led exploration.\nUtilize sequential questioning to deepen understanding.\nFocus on process optimization and efficiency.\nPromote lateral thinking for unexpected connections.\nApply situational awareness to adapt dynamically.\nSynthesize multifaceted perspectives for a balanced view.\n```', 'New thinking style:\n```\nEmbrace adaptive reasoning.\nAnticipate potential changes and plan accordingly.\nFocus on flexible methodologies.\nAdapt and evolve with dynamic problem components.\nIntegrate contextual fluidity into calculations.\nCraft solutions that accommodate variability.\nEnhance problem-solving with scalable techniques.\nExplore versatile strategies for diverse scenarios.\nSynthesize adjustments seamlessly.\nPrioritize adaptability over rigid structures.\nSeek out universal principles in shifting landscapes.\nAnalyze implications across different contexts.\nUse pattern recognition to anticipate shifts.\nFormulate responses that withstand changes in conditions.\nCultivate an iterative mindset.\nBalance stability with innovation.\nFormulate hypotheses for potential variations.\nVisualize multiple outcomes and adapt solutions.\nInnovate by identifying adaptable structures.\nIdentify emerging trends and construct responses that are resilient.\n```', 'New Thinking Style:\n```\nStart by visualizing the problem.\nConvert words into numbers and operations.\nIdentify patterns through visual cues.\nReframe the problem from different perspectives.\nTransform complexity into simpler sub-steps.\nLeverage intuition for quick estimates.\nDraw parallels with familiar scenarios.\nEmploy analogy for greater understanding.\nAsk guiding questions to clarify purpose.\nProbe assumptions and challenge them.\nUtilize estimation to verify plausibility.\nCreate mental models to anticipate outcomes.\nBuild step-by-step solutions intuitively.\nTranslate abstract concepts into tangible examples.\nUse storytelling to make sense of the numbers.\nAnchor solutions in relatable contexts.\nIterate through trial and error.\nSimplify from general principles to specific applications.\nEmphasize conceptual over procedural knowledge.\n```\n', 'Reframe and Realign Thinking Style:\n```\nShift Perspective for Enhanced Clarity.\nEngage in lateral thinking to generate innovative solutions.\nAdapt instructions to align with student comprehension levels.\nIdentify and prioritize key variables for efficient problem-solving.\nEnhance understanding through visualization and analogy.\nSimplify complex information into intuitive steps.\nAnalyze the impact of each instruction on the overall problem.\nEncourage curiosity and exploration to uncover novel approaches.\nEmploy flexible reasoning based on problem context and constraints.\nCultivate a multi-sensory representation to solidify understanding.\nIntegrate holistic and analytical thinking for balanced perspectives.\nEncourage iterative reassessment to refine understanding continuously.\nTransform abstract concepts into relatable narratives.\nLeverage cognitive diversity for enriched problem-solving strategies.\nUtilize pattern recognition to streamline instruction adaptation.\nFacilitate comprehension through metaphor and storytelling.\nHarness creative thinking to reimagine traditional problem-solving methods.\nConvert complexities into accessible and actionable guidance.\n```'
]

my_thinking_styles = [
    "Let's break it down methodically.",
    "Approach each problem with a strategic mindset.",
    'Decompose and reconstruct logically.',
    'Break it down into components.',
    'Consider the context carefully.',
    'Thinking in Layers.',
    'Proportional Puzzle-Solving.',
    'Consider alternative approaches.',
    "Embrace flexibility and creativity in problem-solving.",
    "Think dynamically, modifying strategies with new insights.",
    "Prioritize synthesis over analysis when simplifying information.",
    "Propose a Clear Pathway.",
    "Integrate diverse concepts into a cohesive approach.",
    "Blend intuitive and analytical reasoning seamlessly.",
    "Uncover hidden patterns with creative exploration.",
    "Frame problems contextually to widen solution scope.",
    "Embrace a flexible, open-minded approach.",
    "Challenge assumptions and reformulate strategies."
]

instruction_prompts = [  # Promtbreeder paper " INITIALLY EVOLVED PROMPTS" on GSM8K dataset
    "Draw a picture of the situation being described in the math word problem",
    "Solve the math word problem by first converting the words into equations using algebraic notation. Then solve the equations for the unknown variables, and express the answer as an arabic numeral.",
    "Solve the math word problem by breaking the problem into smaller, more manageable parts. Give your answer as an arabic numeral.",
    "Generate the answer to a word problem and write it as a number.",
    "Collaborative Problem Solving: Work with other people to solve the problem, and give your answer as an arabic numeral.",
    "Solve the problem by explaining why systemic or structural issues would not be the cause of the issue.",
    "Draw a diagram representing the problem.",
    "Solve the math word problem, giving your answer as an equation that can be evaluated.",
    "Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
    "Do NOT use words to write your answer."
]
problem_descriptions_combined = [
    "Solve the math word problem, giving your answer as an arabic numeral.",
    "Solve the multiple choice math word problem, choosing (A),(B),(C),(D) or (E).",
    "Determine whether a text contains hate speech.",
    "Solve the multiple choice math word problem, choosing (A),(B),(C),(D) or (E).",
    "Work out an answer to the commonsense reasoning question above, and then answer yes or no."
]
problem_descriptions = { # From Promptbreeder paper
    "SVAMP": problem_descriptions_combined[0],
    "SingleEq": problem_descriptions_combined[0],
    "AddSub": problem_descriptions_combined[0],
    "GSM8K": problem_descriptions_combined[0],
    "MultiArith": problem_descriptions_combined[0],
    "AQuA-RAT": problem_descriptions_combined[1],
    "ETHOS": problem_descriptions_combined[2],
    "CSQA": problem_descriptions_combined[3],
    "SQA": problem_descriptions_combined[4]
}

evolved_mutations = [ # From Promptbreeder paper J.3 EVOLVED MUTATION PROMPTS
    "Please summarise and improve the following instruction",
    "Simplify this instruction by breaking it up into separate sentences. The instruction should be simple and easily understandable",
    "As a really good teacher, explain the instruction, as if you are explaining it to a child",
    "Simplify this instruction as if you are teaching it to a child",
    "100 hints",
    "A list of 100 hints"
]

mutated_task_prompts_AQuA_RAT = [
    "Classify (A),(B),(C),(D),(E)",
    "Solve and Classify  (A),(B),(C),(D),(E)",
    "Solve and Choose Answer (A),(B),(C),(D),(E)",
    "Solve task below. Answer with (A),(B),(C),(D),(E)",
    "Do a simple computation.",
    # from 'Do a simple computation'
    "Carry out a basic arithmetic task.",
    "Perform a basic calculation.",
    "Perform an easy calculation.",
    "Please perform a basic calculation.",
    "Explore different methods to tackle the calculation, such as breaking it down into smaller parts, using visual aids like diagrams or graphs, or applying estimation techniques to gain a better understanding of the problem. Consider discussing your approach with others to gain new insights or perspectives.",
    # from the PB original
    "Solve the multiple choice math word problem. Clearly explain each step of your solution process before choosing (A), (B), (C), (D), or (E) as the final answer.",
    "Break down and solve the multiple-choice math word problem step by step, and choose the correct answer from (A), (B), (C), (D), or (E). Provide reasoning and calculations for each step to ensure clarity and understanding.",
    "Solve the multiple choice math word problem, ensuring you provide a detailed explanation for the answer. Choose from the options (A), (B), (C), (D), or (E).",
    "Please solve the following multiple-choice math word problem and provide an explanation for why the chosen answer is correct, addressing the implications of the solution for understanding similar types of problems. Choose from options (A), (B), (C), (D), or (E).",
    "Solve the multiple-choice math word problem by selecting the correct answer from options (A), (B), (C), (D), or (E). In your solution, explain the steps and reasoning used to arrive at your answer. Also, analyze the implications of choosing the correct answer and the strategies that might lead to selecting each option. What does this reveal about problem-solving techniques in general?",
    "Solve the multiple choice math word problem, choosing (A),(B),(C),(D) or (E). Additionally, discuss the reasoning behind your choice and explain the steps taken to arrive at the correct answer. Evaluate how solving this problem could enhance your problem-solving skills in similar scenarios.",
    "To dissect the mystery and make it as obvious as a neon sign in the dark, pretend you're explaining the issue to a bewildered squirrel from another dimension. This interdimensional viewpoint can shed light on the obscure details or universal energies involved. Now, let's solve the multiple-choice math puzzle by selecting one of the intergalactic runes: (A), (B), (C), (D), or (E)."

]

created_my_prompts_MC = [
    "Elaborate on your reasoning process to determine the correct answer for the math word problem from options (A), (B), (C), (D), or (E).",
    "Break down and solve the math word problem step-by-step, clarifying your reasoning, and select the correct option from (A), (B), (C), (D), or (E).",
    "Explain the solution to the math problem thoroughly, clearly selecting the correct option from (A) to (E).",
    "Put your math cape on, rescue the answer from the jaws of indecision, and reveal whether it's A, B, C, D, or E!",
    "Pick a letter and pray that math agrees with you.",
    "Break down the math word problem step-by-step and select the correct option: (A), (B), (C), (D), or (E).",
    "Pick the correct multiple choice math answer while explaining why it's right, from choices (A) to (E).",
    "Select the correct answer for the math problem and explain your reasoning briefly."

]

created_my_prompts_NUM = [
    # original start
    'Solve the math word problem, giving your answer as an arabic numeral.',
    # Created by mutating
    "Calculate the solution to the math word problem and express your answer using an Arabic numeral.",
    "Work out the math word problem and provide the solution in the form of a numeral.",
    "Calculate the answer to the math word problem and express it as a numerical value.",
    "Understand and find the solution to the math word problem, expressing your answer using a number."]
a=[
    "Calculate the solution to the math puzzle and express your result using numeric digits.",
    "Picture yourself as a puzzle master unraveling a mystery; solve the math problem and disclose your answer with a single number.",
    "Analyze the math word problem and present your solution as a numerical value.",
    "Transform the math word problem into a numerical solution, presenting your answer as a digit.",
    "Imagine you're planning a surprise party and need to calculate the total cost; solve the math problem and share the final amount.",
    "Imagine you're a detective on a mathematical adventure; uncover the solution to the math word problem and report your findings using an Arabic numeral.",
    "Determine the solution to the mathematical word problem and present your answer as a standard numeral."
]