class StageMachine:
    def __init__(self):
        self.stages = [
            "PARSE_INTENT",
            "DATA_SOURCE",
            "PROFILE_DATA",
            "PREPROCESS",
            "MODEL_SELECT",
            "TRAIN",
            "REVIEW_EDIT",
            "EXPORT"
        ]
        self.current_stage_index = 0

    def get_current_stage(self):
        return self.stages[self.current_stage_index]

    def advance_stage(self):
        if self.current_stage_index < len(self.stages) - 1:
            self.current_stage_index += 1
            return self.get_current_stage()
        else:
            raise Exception("No more stages to advance to.")

    def reset(self):
        self.current_stage_index = 0

    def is_stage_complete(self):
        # Placeholder for actual completion logic
        return True

    def confirm_stage(self):
        if self.is_stage_complete():
            return self.advance_stage()
        else:
            raise Exception("Current stage is not complete.")