def create_workout_plan(age, gender, lifestyle):
    """Generate a workout plan based on user input."""
    # Simplified logic for demonstration purposes
    plan = f"As a {age}-year-old {gender} with a {lifestyle} lifestyle, your workout plan should include:\n"
    if lifestyle.lower() == "active":
        plan += "- High-intensity interval training (HIIT)\n- Strength training 4 times a week"
    else:
        plan += "- Moderate aerobic exercises\n- Light strength training 2 times a week"
    return plan

def provide_diet_tips(age, lifestyle):
    """Provide diet tips based on age and lifestyle."""
    tips = f"For a {age}-year-old with a {lifestyle} lifestyle, consider the following diet tips:\n"
    if lifestyle.lower() == "active":
        tips += "- Increase protein intake\n- Stay hydrated\n- Balance carbs and fats"
    else:
        tips += "- Incorporate more fruits and vegetables\n- Reduce sugar intake\n- Stay hydrated"
    return tips
