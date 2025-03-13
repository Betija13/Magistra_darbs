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
    AnswerType.NUMBER.value: "Single numeric answer:\n",
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
    "Do a simple computation.",
    "Solve and Classify  (A),(B),(C),(D),(E)",
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