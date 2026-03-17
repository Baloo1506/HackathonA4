import React, { useState, useMemo, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
} from 'react-native';
import { predict, EmployeeInput, PredictionResult } from './scorer';

import { API_URL, API_KEY } from './apiConfig';

const DEPARTMENTS = [
  'Admin Offices', 'Executive Office', 'IT/IS',
  'Production', 'Sales', 'Software Engineering',
];

const C = {
  bg: '#0a0d14',
  surface: '#11141c',
  card: '#161b27',
  border: '#1e2638',
  text: '#dde2f0',
  muted: '#5c6580',
  subtleText: '#9199b5',
  accent: '#5b8efa',
  high: '#ef4444',
  medium: '#f59e0b',
  low: '#22c55e',
};

const DEFAULT: EmployeeInput = {
  age: '',
  tenure_years: '',
  salary: '',
  department: 'Production',
  performance_score: 'Fully Meets',
  special_projects: '',
  engagement: 3,
  satisfaction: 3,
  absences: '',
  days_late: '',
  has_transfer_request: false,
  concern_salary: false,
  concern_growth: false,
  concern_workload: false,
  concern_management: false,
  concern_departure: false,
};

/* ── mini composants ─────────────────────────────────── */

function SectionHeader({ title }: { title: string }) {
  return (
    <View style={s.sectionHeader}>
      <Text style={s.sectionTitle}>{title}</Text>
      <View style={s.sectionLine} />
    </View>
  );
}

function NumInput({
  label, placeholder, value, onChange,
}: { label: string; placeholder: string; value: string; onChange: (v: string) => void }) {
  return (
    <View style={s.numField}>
      <Text style={s.fieldLabel}>{label}</Text>
      <TextInput
        style={s.input}
        value={value}
        onChangeText={onChange}
        keyboardType="numeric"
        placeholder={placeholder}
        placeholderTextColor={C.muted}
      />
    </View>
  );
}

function RatingRow({
  label, value, onChange,
}: { label: string; value: number; onChange: (v: number) => void }) {
  const emoji = value <= 2 ? '😟' : value === 3 ? '😐' : '😊';
  return (
    <View style={s.ratingRow}>
      <Text style={s.fieldLabel}>{label}</Text>
      <View style={s.ratingGroup}>
        {[1, 2, 3, 4, 5].map(n => (
          <TouchableOpacity
            key={n}
            style={[s.ratingBtn, value === n && s.ratingBtnOn]}
            onPress={() => onChange(n)}
          >
            <Text style={[s.ratingBtnTxt, value === n && s.ratingBtnTxtOn]}>{n}</Text>
          </TouchableOpacity>
        ))}
      </View>
      <Text style={s.ratingEmoji}>{emoji}</Text>
    </View>
  );
}

function SegmentedRow({
  label, value, options, onChange,
}: { label: string; value: string; options: string[]; onChange: (v: string) => void }) {
  return (
    <View style={s.segCol}>
      <Text style={s.fieldLabel}>{label}</Text>
      <View style={s.segGroup}>
        {options.map(opt => (
          <TouchableOpacity
            key={opt}
            style={[s.segBtn, value === opt && s.segBtnOn]}
            onPress={() => onChange(opt)}
          >
            <Text style={[s.segBtnTxt, value === opt && s.segBtnTxtOn]} numberOfLines={1}>
              {opt}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

function CheckRow({
  label, value, onChange,
}: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <TouchableOpacity style={s.checkRow} onPress={() => onChange(!value)}>
      <View style={[s.checkbox, value && s.checkboxOn]}>
        {value && <Text style={s.checkMark}>✓</Text>}
      </View>
      <Text style={s.checkLabel}>{label}</Text>
    </TouchableOpacity>
  );
}

/* ── jauge de risque ─────────────────────────────────── */

function RiskGauge({ score, level }: { score: number; level: string }) {
  const pct = Math.round(score * 100);
  const color = level === 'Élevé' ? C.high : level === 'Moyen' ? C.medium : C.low;
  return (
    <View style={s.gaugeWrap}>
      <View style={s.gaugeRow}>
        <Text style={[s.gaugeScore, { color }]}>{pct}%</Text>
        <View style={[s.riskBadge, { borderColor: color }]}>
          <Text style={[s.riskBadgeTxt, { color }]}>Risque {level}</Text>
        </View>
      </View>
      <View style={s.gaugeBar}>
        <View style={[s.gaugeFill, { width: `${pct}%` as any, backgroundColor: color }]} />
        {/* marqueurs 30% et 60% */}
        <View style={[s.gaugeMarker, { left: '30%' as any }]} />
        <View style={[s.gaugeMarker, { left: '60%' as any }]} />
      </View>
      <View style={s.gaugeTicksRow}>
        <Text style={s.gaugeTick}>Faible</Text>
        <Text style={s.gaugeTick}>Moyen</Text>
        <Text style={s.gaugeTick}>Élevé</Text>
      </View>
    </View>
  );
}

/* ── écran principal ─────────────────────────────────── */

export default function PredictionScreen() {
  const [form, setForm] = useState<EmployeeInput>(DEFAULT);
  const [apiResult, setApiResult] = useState<PredictionResult | null>(null);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [loading, setLoading] = useState(false);

  function set<K extends keyof EmployeeInput>(key: K, val: EmployeeInput[K]) {
    setForm(prev => ({ ...prev, [key]: val }));
  }

  // Check API health on mount
  useEffect(() => {
    const headers: Record<string, string> | undefined = API_KEY
      ? { 'X-API-Key': API_KEY }
      : undefined;
    fetch(`${API_URL}/health`, {
      signal: AbortSignal.timeout(2000),
      headers,
    })
      .then(r => r.json())
      .then(d => setApiStatus(d.status === 'ok' ? 'online' : 'offline'))
      .catch(() => setApiStatus('offline'));
  }, []);

  // Call real API when form changes
  const callApi = useCallback(async (f: EmployeeInput) => {
    if (apiStatus !== 'online') return;
    setLoading(true);
    try {
      const body = {
        age: parseFloat(f.age) || null,
        tenure_years: parseFloat(f.tenure_years) || null,
        salary: parseFloat(f.salary) || null,
        Department: f.department,
        PerformanceScore: f.performance_score,
        special_projects_count: parseFloat(f.special_projects) || 0,
        engagement_survey: f.engagement,
        emp_satisfaction: f.satisfaction,
        absences: parseFloat(f.absences) || 0,
        days_late_last_30: parseFloat(f.days_late) || 0,
        has_transfer_request: f.has_transfer_request ? 1 : 0,
        transfer_request_sentiment: f.has_transfer_request ? -0.3 : 0,
        feedback_has_compensation: f.concern_salary ? 1 : 0,
        feedback_has_growth: f.concern_growth ? 1 : 0,
        feedback_has_workload: f.concern_workload ? 1 : 0,
        feedback_has_management: f.concern_management ? 1 : 0,
        feedback_has_departure_intent: f.concern_departure ? 1 : 0,
        feedback_sentiment: [f.concern_departure, f.concern_salary, f.concern_management].filter(Boolean).length > 1 ? -0.5 : -0.2,
      };
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: (() => {
          const headers: Record<string, string> = { 'Content-Type': 'application/json' };
          if (API_KEY) headers['X-API-Key'] = API_KEY;
          return headers;
        })(),
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(5000),
      });
      const data = await res.json();
      // Map API response to PredictionResult shape (factors/recommendations from local)
      const local = predict(f);
      setApiResult({ score: data.score, risk_level: data.risk_level, factors: local.factors, recommendations: local.recommendations });
    } catch {
      setApiStatus('offline');
    } finally {
      setLoading(false);
    }
  }, [apiStatus]);

  useEffect(() => { callApi(form); }, [form, callApi]);

  // Use API result if available, else fall back to local scorer
  const localResult = useMemo(() => predict(form), [form]);
  const result = apiResult ?? localResult;

  return (
    <View style={s.root}>
      {/* ── header ── */}
      <View style={s.header}>
        <Text style={s.headerIcon}>🧠</Text>
        <View style={{ flex: 1 }}>
          <Text style={s.headerTitle}>HR Attrition AI</Text>
          <Text style={s.headerSub}>Prédiction du risque de départ volontaire</Text>
        </View>
        <View style={[s.apiPill, apiStatus === 'online' ? s.apiOnline : apiStatus === 'offline' ? s.apiOffline : s.apiChecking]}>
          <Text style={s.apiPillTxt}>
            {apiStatus === 'online' ? '🟢 Modèle ML réel' : apiStatus === 'offline' ? '🔴 Règles locales' : '⏳ Connexion...'}
          </Text>
        </View>
      </View>

      {/* ── corps 2 colonnes ── */}
      <View style={s.body}>

        {/* ─ colonne formulaire ─ */}
        <ScrollView style={s.formCol} contentContainerStyle={s.formContent}>

          <SectionHeader title="👤 Profil" />
          <View style={s.row2}>
            <NumInput label="Âge" placeholder="ex : 32" value={form.age} onChange={v => set('age', v)} />
            <NumInput label="Ancienneté (années)" placeholder="ex : 3.5" value={form.tenure_years} onChange={v => set('tenure_years', v)} />
          </View>
          <View style={s.row2}>
            <NumInput label="Salaire ($/an)" placeholder="ex : 65 000" value={form.salary} onChange={v => set('salary', v)} />
            <NumInput label="Projets spéciaux" placeholder="ex : 2" value={form.special_projects} onChange={v => set('special_projects', v)} />
          </View>
          <SegmentedRow
            label="Département"
            value={form.department}
            options={DEPARTMENTS}
            onChange={v => set('department', v)}
          />

          <SectionHeader title="📊 Performance" />
          <SegmentedRow
            label="Score de performance"
            value={form.performance_score}
            options={['PIP', 'Needs Improvement', 'Fully Meets', 'Exceeds']}
            onChange={v => set('performance_score', v)}
          />

          <SectionHeader title="💬 Engagement & Comportement" />
          <RatingRow label="Score d'engagement  (1 = très faible, 5 = excellent)" value={form.engagement} onChange={v => set('engagement', v)} />
          <RatingRow label="Satisfaction employé (1 = très faible, 5 = excellent)" value={form.satisfaction} onChange={v => set('satisfaction', v)} />
          <View style={s.row2}>
            <NumInput label="Absences (jours/an)" placeholder="ex : 8" value={form.absences} onChange={v => set('absences', v)} />
            <NumInput label="Retards (jours/30j)" placeholder="ex : 2" value={form.days_late} onChange={v => set('days_late', v)} />
          </View>

          <SectionHeader title="🚨 Signaux RH" />
          <CheckRow label="Demande de mobilité / transfert interne" value={form.has_transfer_request} onChange={v => set('has_transfer_request', v)} />
          <CheckRow label="Préoccupations salariales exprimées" value={form.concern_salary} onChange={v => set('concern_salary', v)} />
          <CheckRow label="Manque de perspectives d'évolution" value={form.concern_growth} onChange={v => set('concern_growth', v)} />
          <CheckRow label="Surcharge de travail / burnout" value={form.concern_workload} onChange={v => set('concern_workload', v)} />
          <CheckRow label="Problèmes avec le management" value={form.concern_management} onChange={v => set('concern_management', v)} />
          <CheckRow label="Intention de départ exprimée" value={form.concern_departure} onChange={v => set('concern_departure', v)} />

          <View style={{ height: 60 }} />
        </ScrollView>

        {/* ─ colonne résultats ─ */}
        <ScrollView style={s.resultCol} contentContainerStyle={s.resultContent}>
          <View style={s.resultTopRow}>
            <Text style={s.resultHeading}>Résultat en temps réel</Text>
            {loading && <Text style={s.loadingTxt}>calcul...</Text>}
          </View>

          <RiskGauge score={result.score} level={result.risk_level} />

          {/* facteurs */}
          <View style={s.card}>
            <Text style={s.cardTitle}>Facteurs principaux</Text>
            {result.factors.length === 0 ? (
              <Text style={s.emptyTxt}>Profil équilibré – aucun facteur dominant</Text>
            ) : result.factors.map((f, i) => (
              <View key={i} style={[s.factorRow, i < result.factors.length - 1 && s.factorBorder]}>
                <View style={[s.factorDot, { backgroundColor: f.positive ? C.low : C.high }]} />
                <Text style={s.factorLabel} numberOfLines={2}>{f.label}</Text>
                <Text style={[s.factorImpact, { color: f.positive ? C.low : C.high }]}>
                  {f.positive ? '−' : '+'}{Math.round(f.impact * 100)} pts
                </Text>
              </View>
            ))}
          </View>

          {/* recommandations */}
          <View style={s.card}>
            <Text style={s.cardTitle}>Actions recommandées</Text>
            {result.recommendations.map((r, i) => (
              <View key={i} style={s.recRow}>
                <View style={s.recNum}><Text style={s.recNumTxt}>{i + 1}</Text></View>
                <Text style={s.recTxt}>{r}</Text>
              </View>
            ))}
          </View>

          {/* note éthique */}
          <View style={s.ethicsCard}>
            <Text style={s.ethicsTxt}>
              ℹ️  Ce score est indicatif. Il se base sur des facteurs comportementaux et
              organisationnels. Aucune donnée démographique (sexe, origine, âge seul) n'entre dans
              le calcul.
            </Text>
          </View>

          <View style={{ height: 40 }} />
        </ScrollView>
      </View>
    </View>
  );
}

/* ── styles ──────────────────────────────────────────── */

const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },

  // header
  header: {
    flexDirection: 'row', alignItems: 'center', gap: 14,
    paddingHorizontal: 24, paddingVertical: 18,
    borderBottomWidth: 1, borderBottomColor: C.border,
    backgroundColor: C.surface,
  },
  headerIcon: { fontSize: 30 },
  headerTitle: { fontSize: 20, fontWeight: '700', color: C.text },
  headerSub: { fontSize: 12, color: C.muted, marginTop: 2 },
  apiPill: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20, borderWidth: 1 },
  apiOnline: { backgroundColor: '#052E16', borderColor: '#166534' },
  apiOffline: { backgroundColor: '#2d0e0e', borderColor: '#7f1d1d' },
  apiChecking: { backgroundColor: '#1c1a05', borderColor: '#713f12' },
  apiPillTxt: { fontSize: 12, color: C.subtleText, fontWeight: '600' },
  resultTopRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 },
  resultHeading: { fontSize: 15, fontWeight: '700', color: C.text },
  loadingTxt: { fontSize: 12, color: C.muted, fontStyle: 'italic' },

  // layout
  body: { flex: 1, flexDirection: 'row' },
  formCol: { flex: 1, borderRightWidth: 1, borderRightColor: C.border },
  formContent: { padding: 22, paddingBottom: 40 },
  resultCol: { flex: 1, backgroundColor: C.surface },
  resultContent: { padding: 22, paddingBottom: 40 },

  // sections
  sectionHeader: {
    flexDirection: 'row', alignItems: 'center', gap: 10,
    marginTop: 26, marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 11, fontWeight: '700', color: C.accent,
    letterSpacing: 1, textTransform: 'uppercase',
  },
  sectionLine: { flex: 1, height: 1, backgroundColor: C.border },

  // inputs
  row2: { flexDirection: 'row', gap: 12, marginBottom: 12 },
  numField: { flex: 1 },
  fieldLabel: { fontSize: 12, color: C.subtleText, marginBottom: 6, fontWeight: '500' },
  input: {
    backgroundColor: C.card, borderWidth: 1, borderColor: C.border,
    borderRadius: 8, paddingHorizontal: 12, paddingVertical: 10,
    color: C.text, fontSize: 14,
  },

  // rating
  ratingRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 14 },
  ratingGroup: { flexDirection: 'row', gap: 6 },
  ratingBtn: {
    width: 38, height: 38, borderRadius: 8,
    backgroundColor: C.card, borderWidth: 1, borderColor: C.border,
    alignItems: 'center', justifyContent: 'center',
  },
  ratingBtnOn: { backgroundColor: C.accent, borderColor: C.accent },
  ratingBtnTxt: { fontSize: 14, color: C.muted, fontWeight: '600' },
  ratingBtnTxtOn: { color: '#fff' },
  ratingEmoji: { fontSize: 22 },

  // segmented
  segCol: { marginBottom: 14 },
  segGroup: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 4 },
  segBtn: {
    paddingHorizontal: 14, paddingVertical: 9, borderRadius: 8,
    backgroundColor: C.card, borderWidth: 1, borderColor: C.border,
  },
  segBtnOn: { backgroundColor: C.accent + '28', borderColor: C.accent },
  segBtnTxt: { fontSize: 13, color: C.muted, fontWeight: '500' },
  segBtnTxtOn: { color: C.accent, fontWeight: '700' },

  // checkbox
  checkRow: {
    flexDirection: 'row', alignItems: 'center', gap: 12,
    paddingVertical: 9, paddingHorizontal: 2,
  },
  checkbox: {
    width: 22, height: 22, borderRadius: 6,
    borderWidth: 1.5, borderColor: C.border,
    backgroundColor: C.card, alignItems: 'center', justifyContent: 'center',
  },
  checkboxOn: { backgroundColor: C.accent, borderColor: C.accent },
  checkMark: { color: '#fff', fontSize: 13, fontWeight: '800' },
  checkLabel: { fontSize: 14, color: C.text, flex: 1 },

  // gauge
  gaugeWrap: { marginBottom: 24 },
  gaugeRow: {
    flexDirection: 'row', alignItems: 'center',
    justifyContent: 'space-between', marginBottom: 12,
  },
  gaugeScore: { fontSize: 42, fontWeight: '800' },
  riskBadge: {
    paddingHorizontal: 16, paddingVertical: 8,
    borderRadius: 20, borderWidth: 2,
  },
  riskBadgeTxt: { fontSize: 15, fontWeight: '700' },
  gaugeBar: {
    height: 18, borderRadius: 9,
    backgroundColor: C.card, overflow: 'hidden', position: 'relative',
  },
  gaugeFill: {
    position: 'absolute', top: 0, left: 0, bottom: 0, borderRadius: 9,
  },
  gaugeMarker: {
    position: 'absolute', top: 0, bottom: 0, width: 2,
    backgroundColor: C.bg + 'cc',
  },
  gaugeTicksRow: {
    flexDirection: 'row', justifyContent: 'space-between',
    marginTop: 6, paddingHorizontal: 2,
  },
  gaugeTick: { fontSize: 11, color: C.muted },

  // résultats
  card: {
    backgroundColor: C.card, borderRadius: 12, padding: 16,
    marginBottom: 14, borderWidth: 1, borderColor: C.border,
  },
  cardTitle: {
    fontSize: 11, fontWeight: '700', color: C.muted,
    textTransform: 'uppercase', letterSpacing: 0.8, marginBottom: 14,
  },
  factorRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 9 },
  factorBorder: { borderBottomWidth: 1, borderBottomColor: C.border + '80' },
  factorDot: { width: 8, height: 8, borderRadius: 4, flexShrink: 0 },
  factorLabel: { flex: 1, fontSize: 13, color: C.text },
  factorImpact: { fontSize: 13, fontWeight: '700', flexShrink: 0 },
  emptyTxt: { fontSize: 13, color: C.muted, textAlign: 'center', paddingVertical: 8 },

  recRow: { flexDirection: 'row', gap: 12, paddingVertical: 7, alignItems: 'flex-start' },
  recNum: {
    width: 22, height: 22, borderRadius: 11,
    backgroundColor: C.accent + '33', alignItems: 'center', justifyContent: 'center',
    flexShrink: 0,
  },
  recNumTxt: { color: C.accent, fontSize: 12, fontWeight: '700' },
  recTxt: { flex: 1, fontSize: 13, color: C.text, lineHeight: 20 },

  ethicsCard: {
    backgroundColor: '#0d1a35', borderRadius: 10, padding: 14,
    borderWidth: 1, borderColor: '#1a2e55',
  },
  ethicsTxt: { fontSize: 12, color: '#6a8ec8', lineHeight: 18 },
});
