
def preprocessGeminiOutput(gemini_results):
    student = None
    cat_marks = None

    # Loop through top-level keys
    for key, value in gemini_results.items():
        if "student_details_box" in key.lower():
            student = value["studentdetailsbox"]
        elif "cat_mark_box" in key.lower():
            cat_marks = value["catmarkbox"]

    if not student or not cat_marks:
        raise ValueError("Missing student details or CAT marks in input")

    output = {
        "Name": student.get("name", "Unknown"),
        "Registerno": student.get("register_number", "Unknown"),
        "Marks": {
            str(item["question_number"]): (
                str(item["marks_obtained"]) if item["marks_obtained"] is not None else "None"
            )
            for item in cat_marks
        }
    }
    return output