from dotenv import load_dotenv, find_dotenv

from first_utils import answer_with_ai_1
from second_utils import answer_with_ai_2, validate_with_ai_1


load_dotenv(find_dotenv(), override=True)

def print_array(array):
    print("\n\n" + "=" * 20)
    for elem in array:
        print(elem + "\n" + ("=" * 10))


if __name__ == "__main__":
    print("Starting CV analysis...\n\n" + ("=" * 20) + "\n\n")

    input_resume = "files/Ubajaka_Chijioke_Pastel.pdf"

    questions = [
        "Based on the provided document, showcase Chijioke's skills?",
        "Based on the provided document, describe Chijioke's experience with Python?",
        "Based on the provided document, showcase the candidate's best 5 skills?",
    ]

    answers = answer_with_ai_2(input_resume, questions)
    print_array(answers)

    validation_results =  validate_with_ai_1(input_resume, questions, answers)
    print_array(validation_results)


    print("\n\n" + ("=" * 20) + "\n\nCompleted Resume Analysis!")