# ClassFactory module setup
from pathlib import Path

from pyprojroot.here import here

from src.beamer_bot.BeamerBot import BeamerBot
from src.concept_web.ConceptWeb import ConceptMapBuilder
from src.quiz_maker.QuizMaker import QuizMaker
from src.utils.tools import reset_loggers

reset_loggers()


class ClassFactory:
    def __init__(self, lesson_no, syllabus_path, reading_dir, slide_dir, llm,
                 project_dir=None, output_dir=None, lesson_range=None, **kwargs):
        self.lesson_no = lesson_no
        self.syllabus_path = syllabus_path
        self.reading_dir = reading_dir
        self.slide_dir = slide_dir
        self.llm = llm
        self.project_dir = Path(project_dir) if project_dir else here()
        self.output_dir = Path(output_dir) if output_dir else here() / "ClassFactoryOutput"
        self.lesson_range = lesson_range if lesson_range else range(lesson_no, lesson_no + 1)  # Default to a single lesson

    def create_module(self, module_name, **kwargs):
        interim_output_dir = kwargs.get("output_dir", self.output_dir)
        if module_name in ["BeamerBot", "beamerbot"]:
            beamer_output_dir = interim_output_dir / f"BeamerBot/L{self.lesson_no}"
            beamer_output_dir.mkdir(parents=True, exist_ok=True)
            # BeamerBot should still use a single lesson (lesson_no)
            return BeamerBot(
                lesson_no=self.lesson_no,
                syllabus_path=self.syllabus_path,
                reading_dir=self.reading_dir,
                slide_dir=self.slide_dir,
                llm=self.llm,
                output_dir=beamer_output_dir,
                verbose=kwargs.get("verbose", False),
            )
        elif module_name in ["ConceptWeb", "conceptweb"]:
            concept_output_dir = interim_output_dir / f"ConceptWeb/L{self.lesson_no}"
            concept_output_dir.mkdir(parents=True, exist_ok=True)
            # ConceptMapBuilder uses the lesson range
            return ConceptMapBuilder(
                lesson_range=kwargs.get("lesson_range", self.lesson_range),
                readings_dir=self.reading_dir,
                syllabus_path=self.syllabus_path,
                llm=self.llm,
                project_dir=self.project_dir,
                course_name=kwargs.get("course_name", "Political Science"),
                output_dir=concept_output_dir,  # Allow for custom output_dir
                verbose=kwargs.get("verbose", False),  # Allow additional kwargs like verbosity
                recursive=True  # go down one directory to find the lesson
            )

        elif module_name in ["QuizMaker", "quizmaker"]:
            quiz_output_dir = interim_output_dir / f"QuizMaker/L{self.lesson_no}"
            quiz_output_dir.mkdir(parents=True, exist_ok=True)
            return QuizMaker(
                lesson_range=kwargs.get("lesson_range", self.lesson_range),
                llm=self.llm,
                syllabus_path=self.syllabus_path,
                reading_dir=self.reading_dir,
                output_dir=quiz_output_dir,
                course_name=kwargs.get("course_name", "Political Science"),
                prior_quiz_path=self.project_dir / "data/quizzes",
                verbose=kwargs.get("verbose", False),
            )  # Single lesson by default
        else:
            raise ValueError(f"Module {module_name} not recognized.")


if __name__ == "__main__":
    import os

    from langchain_community.llms import Ollama
    from langchain_openai import ChatOpenAI
    from pyprojroot.here import here
    wd = here()

    OPENAI_KEY = os.getenv('openai_key')
    OPENAI_ORG = os.getenv('openai_org')

    lesson_no = 21

    # Path definitions
    readingDir = Path(os.getenv('readingsDir'))
    slideDir = Path(os.getenv('slideDir'))
    syllabus_path = Path(os.getenv('syllabus_path'))

    input_dir = readingDir / f'L{lesson_no}/'
    beamer_example = slideDir / f'L{lesson_no-1}.tex'
    beamer_output = slideDir / f'L{lesson_no}.tex'

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=OPENAI_KEY,
        organization=OPENAI_ORG,
    )
    # llm = Ollama(
    #     model="llama3.1",
    #     temperature=0.2
    # )

    # Initialize the factory
    factory = ClassFactory(lesson_no=lesson_no,
                           syllabus_path=syllabus_path,
                           reading_dir=readingDir,
                           slide_dir=slideDir,
                           llm=llm,
                           project_dir=wd,
                           lesson_range=range(17, 21),
                           course_name="American Government")
    # %%
    # build slides

    specific_guidance = """
    The lesson should be structured in a way that discusses big picture ideas about political campaigns.
    The slides will, at a minimum, cover the following (using the readings as a reference):
               - How campaigns operate
               - How our party system came to be
               - Individual vote choice
               - The types and determinants of today's polarization​
    """

    beamerbot = factory.create_module("BeamerBot", verbose=False)
    slides = beamerbot.generate_slides()           # specific guidance might make the results more generic
    # beamerbot.save_slides(slides)

    # # build concept map
    # builder = factory.create_module("ConceptWeb",
    #                                 course_name = "American Government")

    # builder.build_concept_map()

    # quizmaker = factory.create_module("QuizMaker")

    # quiz = quizmaker.make_a_quiz()
    # quizmaker.launch_interactive_quiz(quiz_data=quiz, sample_size=7)
