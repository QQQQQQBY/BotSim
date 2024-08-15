from typing import List, Optional, Callable, Dict, Any, Type
from pydantic import BaseModel

class Action(BaseModel):
    name: str
    description: str
    func: Optional[Callable[..., str]]
    input_args_schema: Optional[Dict[str, Any]] = None
    enable: bool = True
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_function(
            cls,
            func: Optional[Callable],
            name: str,  # We keep these required to support backwards compatibility
            description: str,
            input_args_schema: dict,
            # output_args_schema: dict,
            **kwargs: Any,
    ):
        return cls(
            name=name,
            func=func,
            description=description,
            input_args_schema=input_args_schema,
            # output_args_schema=output_args_schema,
            **kwargs,
        )

    def args_check(self, args, args_schema):
        for arg_name, arg_type in args_schema.items():
            if arg_name not in args:
                return f"Missing argument: {arg_name}"
            if not isinstance(args[arg_name], arg_type):
                return f"Invalid argument type for {arg_name}: {type(args[arg_name])}, expected {arg_type}"
        return None 

    def __call__(
            self,
            input_action_args: Dict[str, Any],
            env: Any,
            agent: Any,
    ) -> str:
        """Use the tool."""
        # error_message = self.args_check(input_action_args, self.input_args_schema)
        # if error_message:
        #     return error_message
        # else:
        action_output = self.func(input_action_args, env, agent)
        return action_output

class ActionSpace:
    actions: List[Action]
    action_names: List[str]

    def __repr__(self):
        return [{"name": action.name, "description": action.description, 
                 "input_args": action.input_args_schema} for action in self.actions]
    
