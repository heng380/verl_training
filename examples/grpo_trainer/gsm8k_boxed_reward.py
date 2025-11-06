"""
Custom reward function for GSM8K that supports both #### and \boxed{} formats.
"""

import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution_boxed(solution_str, method="flexible"):
    """Extract solution from response, supporting both #### and \\boxed{} formats.
    
    Args:
        solution_str: The response string from the model
        method: "strict" for #### format only, "flexible" for both formats
        
    Returns:
        tuple: (extracted_answer, is_boxed_format)
    """
    assert method in ["strict", "flexible"]
    
    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]
    
    is_boxed_format = False
    final_answer = None
    
    if method == "strict":
        # Try #### format first
        solutions = re.findall(r"####\s*(\-?[0-9\.\,]+)", solution_str)
        if len(solutions) > 0:
            final_answer = solutions[-1].replace(",", "").replace("$", "")
        else:
            # Try \boxed{} format
            boxed_solutions = re.findall(r"\\boxed\{([^}]+)\}", solution_str)
            if len(boxed_solutions) > 0:
                # Extract number from boxed content
                boxed_content = boxed_solutions[-1]
                numbers = re.findall(r"(\-?[0-9\.\,]+)", boxed_content)
                if len(numbers) > 0:
                    final_answer = numbers[-1].replace(",", "").replace("$", "")
                    is_boxed_format = True
    elif method == "flexible":
        # Try \boxed{} format first (more specific)
        boxed_solutions = re.findall(r"\\boxed\{([^}]+)\}", solution_str)
        if len(boxed_solutions) > 0:
            boxed_content = boxed_solutions[-1]
            numbers = re.findall(r"(\-?[0-9\.\,]+)", boxed_content)
            if len(numbers) > 0:
                final_answer = numbers[-1].replace(",", "").replace("$", "")
                is_boxed_format = True
        
        # If no boxed format found, try #### format
        if final_answer is None:
            solutions = re.findall(r"####\s*(\-?[0-9\.\,]+)", solution_str)
            if len(solutions) > 0:
                final_answer = solutions[-1].replace(",", "").replace("$", "")
        
        # Last resort: find any number in the text
        if final_answer is None:
            answer = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
            if len(answer) > 0:
                invalid_str = ["", "."]
                for final_answer in reversed(answer):
                    if final_answer not in invalid_str:
                        break
    
    return final_answer, is_boxed_format


def compute_score(data_source, solution_str, ground_truth, method="flexible", format_score=0.0, score=1.0, **kwargs):
    """The scoring function for GSM8k supporting both #### and \\boxed{} formats.

    Args:
        data_source: The data source identifier (e.g., "openai/gsm8k")
        solution_str: The solution text from the model
        ground_truth: The ground truth answer
        method: The method to extract the solution, choices are 'strict' and 'flexible'
        format_score: The score for incorrect format (default 0.0)
        score: The score for the correct answer (default 1.0)
        **kwargs: Additional keyword arguments (ignored)
        
    Returns:
        float: The computed score (0.0 for wrong, 1.0 for correct)
    """
    answer, is_boxed = extract_solution_boxed(solution_str=solution_str, method=method)
    
    if answer is None:
        return 0.0
    
    # Normalize both answers for comparison
    answer_clean = answer.replace(",", "").replace("$", "").strip()
    ground_truth_clean = str(ground_truth).replace(",", "").replace("$", "").strip()
    
    if answer_clean == ground_truth_clean:
        return score
    else:
        return format_score

