"""Plain-language explanations shown in hover tooltips across the dashboard (one place to edit)."""

HELP = {
    # ---- header / modes
    "app": "PitWall AI reads a race situation, lets five software agents plan, simulate and check "
           "pit-stop options, then recommends one and saves the decision for later review.",
    "data_source": "Where the race numbers come from. Example scenarios are made-up races. Real races replay "
                   "actual timing data (2019–2026, built in, works offline) up to the lap you choose. FastF1 "
                   "(only shown if installed) downloads a race from F1's servers instead.",
    "model_mode": "The agents' brain. With an AI model (Groq, Gemini or Ollama) the Planner asks it which "
                  "strategies to test and Pit Wall Operations asks it to choose and explain the call. Without "
                  "one, fixed rules do both, so the same input always gives the same answer.",
    "ai_brain": "Which AI model the agents use. The AI proposes and decides; the Python simulator does all "
                "the maths and the Critic's rules check every plan, so the AI can't break a racing rule or "
                "invent a lap time. If the AI fails, the rules take over and the log says so.",
    "tyre_model": "How tyre wear is estimated. Formula: hand-written numbers in pitwall/config.py. ML model: "
                  "learned from about 139,000 real laps (2019–2026) and tested on races it never saw. Safe "
                  "tyre-life limits stay the same either way.",
    "decided_by": "Who picked the final call: the AI model (from the Critic-approved shortlist, within "
                  "2 seconds of the fastest) or the rule (always the fastest valid option).",
    "ml_metrics": "Average error per lap (MAE) on 2025–2026 races the model never saw. Lower is better. "
                  "'No model' always guesses the driver's typical lap; 'Formula' is PitWall's hand-written tyre model.",
    "ml_importance": "How much worse the model gets when one input is shuffled at random. Bigger = the "
                     "model relies on it more.",
    "ml_curves": "Seconds slower than a new soft tyre, by tyre age, as learned from real laps (mid-race, "
                 "clean air). Dashed part = beyond the tyre ages seen in the data.",
    "ml_scatter": "Each dot is a real lap from an unseen race: what really happened (across) vs what the "
                  "model predicted (up). Dots near the diagonal are good predictions.",
    "ml_try": "Ask the trained model about any lap: it predicts how much slower or faster than the driver's "
              "typical lap it would be.",
    "eval": "The agents were run at about 400 real race moments (P1, P5 and P10 finishers, 3 laps before "
            "their first real pit stop) and their call compared with the team's.",
    "eval_clean": "Median error of the predicted time to the flag on races with no Safety Car, VSC, red "
                  "flag or rain after the decision (those can't be predicted).",
    "sources": "Real races: 162 races built into the project (works offline). FastF1: a Python library "
               "that downloads timing from F1's servers while the app runs (needs internet and "
               "pip install fastf1). Both are real timing data; the built-in races are a saved copy.",
    "decision_point": "The moment in the race the decision is made. 'After lap 20' means 20 laps are "
                      "finished and the next lap is lap 21.",

    # ---- agents
    "agents": "Five agents pass a shared workspace along a fixed route. The Critic can send work back to "
              "the Planner or Executor up to two times before a decision is made.",
    "agent_planner": "Planner: reads the question and race state, decides which checks to run and which "
                     "pit-stop options to test (with an AI model, the model writes this plan). When the Critic "
                     "objects, it revises the plan.",
    "agent_research": "Race Research: studies tyre wear, gaps to other cars, weather and Safety Car status, "
                      "running only the analyses the Planner asked for.",
    "agent_executor": "Strategy Executor: runs the lap-by-lap race simulation for every option and returns "
                      "the estimated time to the finish.",
    "agent_critic": "Strategy Critic: rejects options that break a rule (tyres too old, pit lane closed, "
                    "no sets left, slick tyres in the rain…) and double-checks every number. It asks for a "
                    "revision when nothing valid is left.",
    "agent_ops": "Pit Wall Operations: makes the call. With an AI model, the model chooses from the "
                 "Critic-approved options within 2 s of the fastest and explains why; otherwise the fastest valid "
                 "option wins. Writes the radio call and saves a decision record automatically.",

    # ---- controls
    "question": "Changes what the Planner tests. 'Best strategy' tries everything; 'pit now or stay out' "
                "tests only those two; 'which compound' compares tyre types for a stop this lap.",
    "analyze": "Runs the full agent workflow on the current inputs and saves the result to the history.",
    "scenario": "Pre-built races: a normal dry race, a Safety Car, rain on the way, and unusually fast "
                "tyre wear. Loading one replaces the inputs below.",

    # ---- race summary tiles
    "lap": "Laps completed at the decision point out of the full race distance. The bar shows race progress.",
    "remaining": "Laps still to race. The simulator models every one of these laps.",
    "track": "Track surface right now. Wet tracks make slick tyres slow and unsafe.",
    "sc": "Safety Car (SC) or Virtual Safety Car (VSC) laps, e.g. SC L21–24. +1 means one more period. Stops "
          "are cheaper then: about half the usual time under a Safety Car, about 60% under a VSC.",
    "rain": "Simulated rain forecast. L26+ means wet from lap 26 to the finish; L26–40 means wet on laps 26 to 40. Never real recorded weather.",
    "deg": "How fast tyres wear compared with normal. 1.5× means 50% more time lost per lap of tyre age, "
           "and a shorter safe tyre life.",
    "tyre": "The target car's current tyre type and how many laps it has already done.",

    # ---- events
    "events": "Apply a race event to see how the recommendation reacts. The event updates the inputs and "
              "re-runs every agent at the same decision point.",
    "ev_sc": "Puts a Safety Car on track for the chosen laps. Stops during it are cheaper.",
    "ev_vsc": "Adds a Virtual Safety Car: everyone slows to a set pace without a pace car. A stop costs about "
              "60% of the normal pit loss.",
    "in_periods": "Add as many extra Safety Car, VSC or rain periods as you like (first and last lap). Use this "
                  "for races with several incidents. Click + to add a row.",
    "ev_rain": "Adds a forecast that the track turns wet. Slick tyres become unsafe on wet laps.",
    "ev_deg": "Multiplies tyre wear. Old tyres slow down faster and reach their safe limit sooner.",

    # ---- recommendation
    "call": "The recommended action, chosen as the fastest option that passed every Critic check.",
    "remaining_time": "Estimated time to finish the race from now: all remaining lap times plus time "
                      "spent in the pit lane.",
    "margin": "How much faster the recommendation is than the next-best valid option. Small margins mean "
              "the choice is close.",
    "pit_loss": "Total time lost driving through the pit lane and changing tyres, counted once per stop.",
    "position": "Where the car would finish among the cars included in this simulation, by total time. "
                "Overtaking isn't modelled.",
    "risk": "A fixed rule, not a probability. High: margin under 1.5 s or the last set of tyres is near its "
            "limit. Medium: margin under 5 s, it's the only valid option, or it depends on rain/Safety Car "
            "timing. Otherwise low.",
    "why": "The calculated facts behind the call, straight from the simulation.",
    "tradeoffs": "The best option of each other type and how much slower it is — or why the Critic rejected it.",
    "radio": "A made-up radio message written from the recommendation, to show how the call would be "
             "passed to the driver. It is not a real communication.",
    "compare": "Compares this analysis with the previous one at the same lap. If the call changed, it "
               "shows the numbers that caused it.",

    "real_events": "Hindsight mode for real races. ON: the agents are told when the real Safety Cars, Virtual "
                   "Safety Cars and rain happened later in the race, so you can see what they'd have done with "
                   "perfect knowledge. OFF (default): they only know what the team knew at the decision lap.",
    "live": "A live view of the run. Every line is a real step from the workflow log, shown as it happens: "
            "plans, tool runs, checks and revisions.",
    "pipeline": "The agents work in order and pass results along: Planner → Race Research → Strategy Executor → "
                "Strategy Critic → Pit Wall Operations. The Critic can send work back to the Planner or Executor.",
    "live_toggle": "On: when you press Analyze Strategy, watch each agent work step by step (slightly slowed so you "
                   "can follow). Off: run instantly.",
    # ---- real race check
    "reality": "Hindsight check for real races. After the agents decide, the app reads the rest of the real race "
               "and compares: did the team do the same thing, how good was the model's time estimate, and what "
               "happened that nobody could know in advance (Safety Cars, rain).",
    "real_time": "How long the driver really took from the decision lap to the chequered flag.",
    "model_est": "What the model predicted for the strategy the team actually ran, using only information "
                 "from before the decision point.",
    "model_error": "Real time minus the model's estimate for the same strategy. Positive = reality was slower "
                   "(often a Safety Car or a slow stop the model didn't know about).",
    "model_gain": "By the model's own numbers, how much faster (+) or slower (−) the agents' call was than the "
                  "team's real strategy. This is a model estimate, not a fact.",
    "chart_reality": "Each bar is one lap after the decision: real lap time minus the model's estimate for the "
                     "team's real strategy. Orange laps were under a Safety Car, VSC or red flag.",
    "chat": "A chat assistant connected to the agents. Ask about the current call, or describe a scenario (lap, "
            "tyres, Safety Cars, VSCs, rain, tyre wear, pit-stop time): it runs all five agents on it and explains "
            "the outcome in plain English with a chart. With an AI model it understands free wording and explains simply; without one it uses fixed rules.",

    # ---- charts
    "chart_timeline": "Each row is a plan from lap now to the flag. Bar colours are tyre types; 'box L24' marks a pit "
                      "stop. Shaded areas are the Safety Car (orange), VSC (amber) and rain (blue) in the scenario.",
    "chart_gap": "Each bar is one option. Length = how many seconds slower it is than the recommendation "
                 "over the rest of the race. Red bars broke a rule and can't be chosen even if fast.",
    "chart_trace": "Running time gap to the recommended plan, lap by lap. Above zero = behind it. Jumps "
                   "are pit stops; slopes show tyre and weather effects.",
    "chart_laps": "Every simulated lap of the recommended plan, coloured by tyre type. Tall bars are "
                  "pit laps, Safety Car laps or wet laps on the wrong tyres.",

    # ---- tables
    "t_verdict": "What the Critic decided. Recommended = chosen. Valid = passed every check. "
                 "Rejected = broke at least one rule (see the last column).",
    "t_strategy": "The pit-stop plan. 'Box L24' means pit at the end of lap 24.",
    "t_type": "Which kind of option this is: stop this lap, stop a few laps later, never stop, a stop "
              "timed to an event, or two stops.",
    "t_origin": "Which plan version proposed it. Revisions appear when the Critic sent the plan back.",
    "t_stops": "Number of further pit stops.",
    "t_time": "Estimated time from now to the chequered flag (minutes:seconds).",
    "t_gap": "Seconds slower than the recommendation. Negative values only occur on rejected options.",
    "t_driving": "Time spent lapping, excluding the pit lane.",
    "t_pit": "Time lost in the pit lane, counted once per stop.",
    "t_finish_tyre": "Tyre type and age at the chequered flag.",
    "t_traffic": "Laps lost behind another car after rejoining from the pits.",
    "t_pos": "Projected finishing place within the simulated group of cars.",
    "t_notes": "Why the Critic rejected this option, in plain words.",

    # ---- agents tab
    "plans": "Each plan version the Planner produced. A second version means the Critic rejected the first.",
    "research": "Findings from the analyses the Planner requested. Skipped analyses weren't needed.",
    "reviews": "Every Critic review round: how many options passed, failed, and what it asked to change.",
    "log": "Every real action, in order: plans, tool calls, checks and revisions. Nothing is invented.",
    "rivals": "Other cars are assumed to choose their own fastest valid strategy under the same model. "
              "This is an assumption, not a prediction.",

    # ---- inputs tab
    "in_total": "Full race distance in laps.",
    "in_done": "Laps finished when the decision is made. Must be less than the total.",
    "in_pit": "Seconds lost for a normal green-flag pit stop (pit lane + tyre change).",
    "in_track": "Whether the track is wet right now.",
    "in_deg": "1.0 = normal tyre wear. Higher numbers mean faster wear and shorter tyre life.",
    "in_rain_start": "First lap that will be wet (simulated forecast). 0 = no rain.",
    "in_rain_end": "Last wet lap. 0 = wet until the finish.",
    "in_sc_start": "First Safety Car lap. 0 = no Safety Car.",
    "in_sc_end": "Last Safety Car lap. Needed whenever a start lap is set.",
    "in_closed": "Laps when the pit lane is closed, e.g. 23, 24. Stops on these laps are rejected.",
    "in_compound": "Tyre type currently fitted.",
    "in_age": "Laps already completed on the current tyres.",
    "in_pace": "Lap time on fresh soft tyres in the dry, in seconds. Tyre, weather and Safety Car effects "
               "are added on top.",
    "in_gap": "Seconds behind the race leader. Used to compare finishing order.",
    "in_used": "Tyre types already used this race. Dry races need two different dry types.",
    "in_sets": "Unused tyre sets left. An option needing a set you don't have is rejected.",
    "in_comp": "Other cars in the simulation. Edit gaps, pace or tyres directly in the table.",

    # ---- history / data
    "history": "One record per completed analysis, saved automatically to data/decisions. Re-running the "
               "page never creates duplicates.",
    "provenance": "Where each input value comes from: real data, a value derived from real data, or a "
                  "synthetic assumption.",
    "model": "The simplified model every option is scored with. All parameters live in pitwall/config.py.",
}


def h(key: str) -> str:
    return HELP.get(key, "")
