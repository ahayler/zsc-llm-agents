from pydantic import BaseModel

class BasePolicyModifierConfig(BaseModel):
    pass

class NoPolicyModifierConfig(BasePolicyModifierConfig):
    pass

class RandomAgentsPolicyModifierConfig(BasePolicyModifierConfig):
    pass