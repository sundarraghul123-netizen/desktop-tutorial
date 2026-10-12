from pydantic import BaseModel, Field

class LessonOutline(BaseModel):
    title: str
    objectives: list[str]
    key_topics: list[str]
    estimated_minutes: int = 10

class ModuleOutline(BaseModel):
    title: str
    description: str
    lessons: list[LessonOutline]
    purpose: str = ""
    objectives: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    practice_tasks: list[str] = Field(default_factory=list)
    assessment: str = ""
    completion_criteria: list[str] = Field(default_factory=list)
    estimated_minutes: int = 0

class Outline(BaseModel):
    title: str
    description: str
    audience: str
    difficulty: str
    prerequisites: list[str]
    objectives: list[str]
    modules: list[ModuleOutline]
    estimated_minutes: int

class SourceTopic(BaseModel):
    title: str
    description: str
    timestamp: str

class SourceTopicCatalog(BaseModel):
    topics: list[SourceTopic]

class CoursePlan(BaseModel):
    topics: list[SourceTopic]
    outline: Outline

class Lesson(BaseModel):
    title: str
    introduction: str
    explanation: str
    concepts: list[str]
    examples: list[str]
    applications: list[str]
    terminology: list[str]
    summary: str
    knowledge_checks: list[str]
    why_it_matters: str = ""
    learning_objectives: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    analogy: str = ""
    step_by_step_breakdown: list[str] = Field(default_factory=list)
    visual_representation: str = ""
    common_mistakes: list[str] = Field(default_factory=list)
    guided_practice: list[str] = Field(default_factory=list)
    independent_practice: list[str] = Field(default_factory=list)
    retrieval_questions: list[str] = Field(default_factory=list)
    mastery_check: list[str] = Field(default_factory=list)
    next_lesson_connection: str = ""

class LessonBatchItem(BaseModel):
    outline_title: str
    lesson: Lesson

class LessonBatch(BaseModel):
    lessons: list[LessonBatchItem]

class MCQ(BaseModel):
    prompt: str
    options: list[str]
    answer_index: int
    explanation: str
    topic: str
    difficulty: str  # easy | medium | hard

class QA(BaseModel):
    qtype: str  # short | long | scenario | interview
    prompt: str
    suggested_answer: str
    marking_points: list[str]
    topic: str
    difficulty: str

class Card(BaseModel):
    front: str
    back: str
    topic: str
    difficulty: str

class Assessment(BaseModel):
    mcqs: list[MCQ]
    qas: list[QA]
    cards: list[Card]
