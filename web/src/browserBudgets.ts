/** Hard caps include complete reference/mutant calibration; progress cannot extend them. */
export function browserBudget(problem: {id: string}) {
  return problem.id === 'diffusion/flow_matching'
    ? {calibration:240_000,run:60_000,total:300_000}
    : {calibration:90_000,run:30_000,total:120_000};
}
