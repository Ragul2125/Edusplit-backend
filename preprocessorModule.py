
def preprocessGeminiOutput(gemini_results,MODE_OF_EXAM):
    student = None
    cat_marks = None
    model_marks=None
     
    if MODE_OF_EXAM == "cat":
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

    elif MODE_OF_EXAM =="model":
                # Loop through top-level keys
        for key, value in gemini_results.items():
            if "student_details_box" in key.lower():
                student = value["studentdetailsbox"]
            elif "cat_mark_box" in key.lower():
                model_marks = value["modelmarkbox"]

        if not student or not model_marks:
            raise ValueError("Missing student details or CAT marks in input")

        output = {
            "Name": student.get("name", "Unknown"),
            "Registerno": student.get("register_number", "Unknown"),
            "Marks": {
                str(item["question_number"]): (
                    str(item["marks_obtained"]) if item["marks_obtained"] is not None else "None"
                )
                for item in model_marks
            }
        }
        return output