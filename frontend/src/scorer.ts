export interface EmployeeInput {
  age: string;
  tenure_years: string;
  salary: string;
  department: string;
  performance_score: string;
  special_projects: string;
  engagement: number;
  satisfaction: number;
  absences: string;
  days_late: string;
  has_transfer_request: boolean;
  concern_salary: boolean;
  concern_growth: boolean;
  concern_workload: boolean;
  concern_management: boolean;
  concern_departure: boolean;
}

export interface Factor {
  label: string;
  impact: number;
  positive: boolean;
}

export interface PredictionResult {
  score: number;
  risk_level: 'Faible' | 'Moyen' | 'Élevé';
  factors: Factor[];
  recommendations: string[];
}

export function predict(inp: EmployeeInput): PredictionResult {
  const factors: Factor[] = [];
  let score = 0.15;

  const tenure = parseFloat(inp.tenure_years) || 0;
  const absences = parseFloat(inp.absences) || 0;
  const daysLate = parseFloat(inp.days_late) || 0;
  const projects = parseFloat(inp.special_projects) || 0;

  // Engagement
  if (inp.engagement <= 2) {
    score += 0.14;
    factors.push({ label: "Engagement très faible", impact: 0.14, positive: false });
  } else if (inp.engagement === 3) {
    score += 0.04;
    factors.push({ label: "Engagement moyen", impact: 0.04, positive: false });
  } else if (inp.engagement >= 4) {
    score -= 0.06;
    factors.push({ label: "Bon niveau d'engagement", impact: 0.06, positive: true });
  }

  // Satisfaction
  if (inp.satisfaction <= 2) {
    score += 0.13;
    factors.push({ label: "Satisfaction très faible", impact: 0.13, positive: false });
  } else if (inp.satisfaction === 3) {
    score += 0.03;
    factors.push({ label: "Satisfaction moyenne", impact: 0.03, positive: false });
  } else if (inp.satisfaction >= 4) {
    score -= 0.06;
    factors.push({ label: "Bonne satisfaction employé", impact: 0.06, positive: true });
  }

  // Performance
  const perfMap: Record<string, { impact: number; label: string; positive: boolean }> = {
    PIP: { impact: 0.18, label: "Plan d'amélioration (PIP)", positive: false },
    "Needs Improvement": { impact: 0.10, label: "Performance insuffisante", positive: false },
    "Fully Meets": { impact: 0.00, label: "", positive: true },
    Exceeds: { impact: 0.07, label: "Performance excellente", positive: true },
  };
  const perf = perfMap[inp.performance_score];
  if (perf && perf.label) {
    score += perf.positive ? -perf.impact : perf.impact;
    factors.push({ label: perf.label, impact: perf.impact, positive: perf.positive });
  }

  // Signaux RH / NLP
  if (inp.concern_departure) {
    score += 0.18;
    factors.push({ label: "Intention de départ exprimée", impact: 0.18, positive: false });
  }
  if (inp.has_transfer_request) {
    score += 0.11;
    factors.push({ label: "Demande de mobilité interne", impact: 0.11, positive: false });
  }
  if (inp.concern_growth) {
    score += 0.08;
    factors.push({ label: "Manque de perspectives d'évolution", impact: 0.08, positive: false });
  }
  if (inp.concern_salary) {
    score += 0.08;
    factors.push({ label: "Préoccupations salariales", impact: 0.08, positive: false });
  }
  if (inp.concern_management) {
    score += 0.07;
    factors.push({ label: "Problèmes avec le management", impact: 0.07, positive: false });
  }
  if (inp.concern_workload) {
    score += 0.06;
    factors.push({ label: "Surcharge de travail", impact: 0.06, positive: false });
  }

  // Absences
  if (absences > 15) {
    const imp = Math.min((absences - 15) * 0.008, 0.09);
    score += imp;
    factors.push({ label: `Absentéisme élevé (${absences}j)`, impact: imp, positive: false });
  } else if (absences >= 0 && absences <= 3) {
    score -= 0.02;
    factors.push({ label: "Très faible absentéisme", impact: 0.02, positive: true });
  }

  // Days late
  if (daysLate > 3) {
    const imp = Math.min(daysLate * 0.015, 0.08);
    score += imp;
    factors.push({ label: `Retards fréquents (${daysLate}j/30j)`, impact: imp, positive: false });
  }

  // Ancienneté
  if (tenure > 0 && tenure < 1) {
    score += 0.07;
    factors.push({ label: "Très courte ancienneté", impact: 0.07, positive: false });
  } else if (tenure >= 3 && tenure <= 5) {
    score += 0.04;
    factors.push({ label: "Ancienneté à risque (3-5 ans)", impact: 0.04, positive: false });
  } else if (tenure > 8) {
    score -= 0.05;
    factors.push({ label: "Forte ancienneté (fidélité)", impact: 0.05, positive: true });
  }

  // Projets spéciaux
  if (projects >= 3) {
    score -= 0.04;
    factors.push({ label: "Forte implication dans les projets", impact: 0.04, positive: true });
  }

  score = Math.max(0.02, Math.min(0.97, score));
  factors.sort((a, b) => b.impact - a.impact);
  const top3 = factors.slice(0, 3);

  const risk_level: PredictionResult["risk_level"] =
    score >= 0.6 ? "Élevé" : score >= 0.3 ? "Moyen" : "Faible";

  const recommendations: string[] = [];
  if (inp.satisfaction <= 2) recommendations.push("Planifier un entretien individuel de satisfaction RH");
  if (inp.engagement <= 2) recommendations.push("Discussion avec le manager – réaffectation ou enrichissement de poste");
  if (inp.concern_departure) recommendations.push("Entretien de rétention urgent avec la DRH");
  if (inp.has_transfer_request) recommendations.push("Initier un entretien de mobilité interne");
  if (inp.concern_salary) recommendations.push("Étude de rémunération comparée à la médiane du département");
  if (inp.concern_growth) recommendations.push("Élaborer un plan de développement et parcours de promotion");
  if (inp.concern_workload) recommendations.push("Audit de la charge de travail et renforcement d'équipe");
  if (inp.concern_management) recommendations.push("Coaching managérial ou réorganisation d'équipe");
  if (absences > 15 || daysLate > 5) recommendations.push("Entretien bien-être et solutions de flexibilité");
  if (inp.performance_score === "PIP" || inp.performance_score === "Needs Improvement") {
    recommendations.push("Plan d'amélioration des performances avec suivi mensuel");
  }
  if (recommendations.length === 0) recommendations.push("Maintenir le suivi RH régulier – profil stable");

  return { score, risk_level, factors: top3, recommendations };
}
