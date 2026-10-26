## v2.0.0 - 2026-10-26
#### Features
- <span style="background-color: #d73a49; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 0.85em;">BREAKING</span>introduce provider clients, including new TypeSafeAi client, and global LLMClient entrypoint - (283d260) - lucaselias61
#### Miscellaneous Chores
- (**version**) v2.0.0 - (d3edb12) - lucaselias61
- correct version bump hook paths - (e1fe2d8) - lucaselias61
- pin new python version and update hooks - (5d71470) - lucaselias61
- add CHANGELOG - (90099c1) - lucaselias61

- - -
## v2.1.0 - 2026-10-26
#### Features
- default temperature and per-query overrides in LLMClient - (1d9224d) - lucaselias61
#### Documentation
- regenerate changelog from commit history - (d3a5a19) - lucaselias61

- - -


## v1.0.0 - 2026-09-10
#### Bug Fixes
- 'total_tokens' now gets correctly writen to usage - (62104f0) - lucaselias61
- temperature omitted for reasoning models and max_output_tokens threaded through all three provider paths - (3cdaa33) - lucaselias61
- annotations import - (18604ea) - lucaselias61
- broken refactor - (212f374) - lucaselias61
- higher cocncurrency limit - (f483806) - lucaselias61
- cached tokens tag - (1714e22) - lucaselias61
#### Refactoring
- new separate query functions for each provider - (cd2dbcd) - lucaselias61
#### Miscellaneous Chores
- turn project into a downloadable package - (7eeaa7c) - lucaselias61

- - -

## v0.2.0 - 2026-06-10
#### Features
- Add XFA fill configuration and orchestrator for PDF form filling - (1e8f204) - lucaselias61
#### Bug Fixes
- repair pipeline wiring, extraction correctness, and solver validation across the board - (b1d6f83) - lucaselias61
- temperature skip for some models - (ecf9d60) - lucaselias61


