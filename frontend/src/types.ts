export interface Employee {
  anonymized_id: string;
  Department: string;
  Position: string;
  risk_score: number;
  risk_level: 'Low' | 'Medium' | 'High';
  top_reason_1: string;
  top_reason_2: string;
  top_reason_3: string;
  recommended_action: string;
  nlp_advisory: string;
}

export type Screen = 'dashboard' | 'employees' | 'detail';

export const COLORS = {
  bg: '#0F172A',
  surface: '#1E293B',
  surfaceHover: '#263348',
  border: '#334155',
  accent: '#6366F1',
  text: '#F8FAFC',
  textMuted: '#94A3B8',
  textDim: '#475569',
  high: '#EF4444',
  highBg: '#450A0A',
  medium: '#F97316',
  mediumBg: '#431407',
  low: '#22C55E',
  lowBg: '#052E16',
  sidebar: '#020817',
};

export const RISK_COLOR = (level: Employee['risk_level']) => {
  if (level === 'High') return COLORS.high;
  if (level === 'Medium') return COLORS.medium;
  return COLORS.low;
};

export const RISK_BG = (level: Employee['risk_level']) => {
  if (level === 'High') return COLORS.highBg;
  if (level === 'Medium') return COLORS.mediumBg;
  return COLORS.lowBg;
};

export const FEATURE_LABELS: Record<string, string> = {
  EmpSatisfaction: 'Satisfaction employé',
  EngagementSurvey: "Score d'engagement",
  engagement_x_satisfaction: 'Engagement × Satisfaction',
  salary_vs_dept_mean: 'Salaire vs. moyenne dept.',
  Salary: 'Salaire annuel',
  Absences: "Nombre d'absences",
  DaysLateLast30: 'Retards sur 30 jours',
  tenure_years: 'Ancienneté (années)',
  SpecialProjectsCount: 'Projets spéciaux',
  age: 'Âge',
  has_transfer_request: 'Demande de mobilité',
  transfer_request_sentiment: 'Sentiment demande mobilité',
  Department_enc: 'Département',
  Position_enc: 'Poste',
  PerformanceScore_enc: 'Score performance',
  MaritalDesc_enc: 'Situation familiale',
  RecruitmentSource_enc: 'Source recrutement',
};
