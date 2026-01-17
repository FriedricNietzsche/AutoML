class ContextPacket:
    def __init__(self, user_prompt, chat_history, current_stage, decisions, dataset_schema, constraints, selected_models, last_run_results):
        self.user_prompt = user_prompt
        self.chat_history = chat_history
        self.current_stage = current_stage
        self.decisions = decisions
        self.dataset_schema = dataset_schema
        self.constraints = constraints
        self.selected_models = selected_models
        self.last_run_results = last_run_results

    def to_dict(self):
        return {
            "user_prompt": self.user_prompt,
            "chat_history": self.chat_history,
            "current_stage": self.current_stage,
            "decisions": self.decisions,
            "dataset_schema": self.dataset_schema,
            "constraints": self.constraints,
            "selected_models": self.selected_models,
            "last_run_results": self.last_run_results,
        }