import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { Employee, COLORS, RISK_COLOR, RISK_BG, FEATURE_LABELS } from '../types';

interface Props {
  employee: Employee;
  onBack: () => void;
}

const RISK_EMOJI: Record<Employee['risk_level'], string> = {
  High: '🔴',
  Medium: '🟠',
  Low: '🟢',
};

const RISK_LABEL: Record<Employee['risk_level'], string> = {
  High: 'Risque ÉLEVÉ',
  Medium: 'Risque MOYEN',
  Low: 'Risque FAIBLE',
};

export default function EmployeeDetailScreen({ employee: emp, onBack }: Props) {
  const riskColor = RISK_COLOR(emp.risk_level);
  const riskBg = RISK_BG(emp.risk_level);
  const scorePct = Math.round(emp.risk_score * 100);

  const reasons = [emp.top_reason_1, emp.top_reason_2, emp.top_reason_3].filter(Boolean);
  const actions = emp.recommended_action
    ? emp.recommended_action.split('|').map(a => a.trim()).filter(Boolean)
    : [];
  const advisories = emp.nlp_advisory
    ? emp.nlp_advisory.split('|').map(a => a.trim()).filter(Boolean)
    : [];

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Back button */}
      <TouchableOpacity style={styles.backBtn} onPress={onBack}>
        <Text style={styles.backText}>← Retour à la liste</Text>
      </TouchableOpacity>

      {/* Hero card */}
      <View style={[styles.heroCard, { borderColor: riskColor }]}>
        <View style={styles.heroTop}>
          <View style={styles.heroInfo}>
            <Text style={styles.heroId}>{emp.anonymized_id}</Text>
            <Text style={styles.heroDept}>{emp.Department}</Text>
            <Text style={styles.heroPosition}>{emp.Position}</Text>
          </View>
          <View style={[styles.heroBadge, { backgroundColor: riskBg, borderColor: riskColor }]}>
            <Text style={styles.heroBadgeEmoji}>{RISK_EMOJI[emp.risk_level]}</Text>
            <Text style={[styles.heroBadgeText, { color: riskColor }]}>
              {RISK_LABEL[emp.risk_level]}
            </Text>
          </View>
        </View>

        {/* Score gauge */}
        <View style={styles.gaugeSection}>
          <View style={styles.gaugeHeader}>
            <Text style={styles.gaugeLabel}>Score de risque de départ</Text>
            <Text style={[styles.gaugeValue, { color: riskColor }]}>{scorePct}%</Text>
          </View>
          <View style={styles.gaugeBg}>
            <View
              style={[
                styles.gaugeFill,
                { width: `${scorePct}%` as any, backgroundColor: riskColor },
              ]}
            />
            {/* Threshold markers */}
            <View style={[styles.gaugeMarker, { left: '30%' as any }]} />
            <View style={[styles.gaugeMarker, { left: '60%' as any }]} />
          </View>
          <View style={styles.gaugeScale}>
            <Text style={styles.gaugeScaleText}>0%</Text>
            <Text style={[styles.gaugeScaleText, { color: COLORS.low }]}>Faible</Text>
            <Text style={[styles.gaugeScaleText, { color: COLORS.medium }]}>Moyen</Text>
            <Text style={[styles.gaugeScaleText, { color: COLORS.high }]}>Élevé</Text>
            <Text style={styles.gaugeScaleText}>100%</Text>
          </View>
        </View>
      </View>

      <View style={styles.row}>
        {/* Risk drivers */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>🎯 Facteurs de risque principaux</Text>
          <Text style={styles.cardSub}>Signaux SHAP identifiés par le modèle IA</Text>
          {reasons.length > 0 ? (
            <View style={styles.driversList}>
              {reasons.map((reason, i) => (
                <View key={i} style={styles.driverItem}>
                  <View style={[styles.driverRank, { backgroundColor: riskBg, borderColor: riskColor }]}>
                    <Text style={[styles.driverRankText, { color: riskColor }]}>{i + 1}</Text>
                  </View>
                  <View style={styles.driverContent}>
                    <Text style={styles.driverName}>
                      {FEATURE_LABELS[reason] ?? reason.replace(/_/g, ' ')}
                    </Text>
                    <Text style={styles.driverRaw}>{reason}</Text>
                  </View>
                </View>
              ))}
            </View>
          ) : (
            <Text style={styles.emptyText}>Aucun facteur significatif — profil stable</Text>
          )}
        </View>

        {/* Recommended actions */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>✅ Actions RH recommandées</Text>
          <Text style={styles.cardSub}>Basées sur les facteurs de risque identifiés</Text>
          {actions.length > 0 ? (
            <View style={styles.actionsList}>
              {actions.map((action, i) => (
                <View key={i} style={styles.actionItem}>
                  <View style={styles.actionBullet}>
                    <Text style={styles.actionBulletText}>{i + 1}</Text>
                  </View>
                  <Text style={styles.actionText}>{action}</Text>
                </View>
              ))}
            </View>
          ) : (
            <Text style={styles.emptyText}>Aucune action spécifique requise</Text>
          )}
        </View>
      </View>

      {/* NLP Advisory */}
      {advisories.length > 0 && (
        <View style={styles.advisoryCard}>
          <View style={styles.advisoryHeader}>
            <Text style={styles.advisoryTitle}>⚠ Signaux NLP additionnels</Text>
            <View style={styles.advisoryTag}>
              <Text style={styles.advisoryTagText}>Hors modèle — indicatif seulement</Text>
            </View>
          </View>
          <Text style={styles.advisoryDesc}>
            Ces signaux proviennent des feedbacks textuels. Ils ne sont pas utilisés dans le calcul
            du score mais peuvent confirmer le risque.
          </Text>
          <View style={styles.advisoryList}>
            {advisories.map((flag, i) => (
              <View key={i} style={styles.advisoryItem}>
                <Text style={styles.advisoryItemIcon}>⚠</Text>
                <Text style={styles.advisoryItemText}>{flag}</Text>
              </View>
            ))}
          </View>
        </View>
      )}

      {/* Ethics note */}
      <View style={styles.ethicsCard}>
        <Text style={styles.ethicsTitle}>🛡 Note éthique</Text>
        <Text style={styles.ethicsText}>
          Ce score est calculé exclusivement à partir de données RH structurées (engagement, satisfaction,
          absences, salaire, ancienneté). Les attributs protégés (sexe, race, origine) ne sont jamais
          utilisés comme variables du modèle. L'ID affiché est anonymisé.
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: 28, gap: 16 },
  backBtn: {
    flexDirection: 'row',
    alignSelf: 'flex-start',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: COLORS.border,
    backgroundColor: COLORS.surface,
  },
  backText: { color: COLORS.textMuted, fontSize: 13, fontWeight: '500' },
  // Hero
  heroCard: {
    backgroundColor: COLORS.surface,
    borderRadius: 12,
    borderWidth: 2,
    padding: 24,
    gap: 20,
  },
  heroTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
  heroInfo: { gap: 4 },
  heroId: { color: COLORS.accent, fontSize: 18, fontWeight: '700' },
  heroDept: { color: COLORS.text, fontSize: 16, fontWeight: '600' },
  heroPosition: { color: COLORS.textMuted, fontSize: 14 },
  heroBadge: {
    alignItems: 'center',
    padding: 16,
    borderRadius: 10,
    borderWidth: 1,
    gap: 6,
  },
  heroBadgeEmoji: { fontSize: 28 },
  heroBadgeText: { fontSize: 13, fontWeight: '700' },
  // Gauge
  gaugeSection: { gap: 8 },
  gaugeHeader: { flexDirection: 'row', justifyContent: 'space-between' },
  gaugeLabel: { color: COLORS.textMuted, fontSize: 13 },
  gaugeValue: { fontSize: 18, fontWeight: '700' },
  gaugeBg: {
    height: 14,
    backgroundColor: COLORS.border,
    borderRadius: 7,
    overflow: 'hidden',
    position: 'relative',
  },
  gaugeFill: { height: '100%', borderRadius: 7 },
  gaugeMarker: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 2,
    backgroundColor: '#0F172A',
  },
  gaugeScale: { flexDirection: 'row', justifyContent: 'space-between' },
  gaugeScaleText: { color: COLORS.textDim, fontSize: 10 },
  // Cards
  row: { flexDirection: 'row', gap: 16 },
  card: {
    flex: 1,
    backgroundColor: COLORS.surface,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 20,
    gap: 10,
  },
  cardTitle: { color: COLORS.text, fontSize: 15, fontWeight: '600' },
  cardSub: { color: COLORS.textDim, fontSize: 12 },
  // Drivers
  driversList: { gap: 10 },
  driverItem: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  driverRank: {
    width: 28,
    height: 28,
    borderRadius: 14,
    borderWidth: 1.5,
    alignItems: 'center',
    justifyContent: 'center',
  },
  driverRankText: { fontSize: 13, fontWeight: '700' },
  driverContent: { flex: 1 },
  driverName: { color: COLORS.text, fontSize: 13, fontWeight: '500' },
  driverRaw: { color: COLORS.textDim, fontSize: 10, marginTop: 1 },
  emptyText: { color: COLORS.textDim, fontSize: 13, fontStyle: 'italic', paddingVertical: 8 },
  // Actions
  actionsList: { gap: 10 },
  actionItem: { flexDirection: 'row', alignItems: 'flex-start', gap: 12 },
  actionBullet: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: '#1E3A2E',
    borderWidth: 1,
    borderColor: COLORS.low,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 1,
  },
  actionBulletText: { color: COLORS.low, fontSize: 11, fontWeight: '700' },
  actionText: { color: COLORS.text, fontSize: 13, flex: 1, lineHeight: 20 },
  // Advisory
  advisoryCard: {
    backgroundColor: '#1C1109',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#92400E',
    padding: 20,
    gap: 10,
  },
  advisoryHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  advisoryTitle: { color: COLORS.medium, fontSize: 15, fontWeight: '600' },
  advisoryTag: {
    backgroundColor: '#431407',
    borderWidth: 1,
    borderColor: '#92400E',
    paddingVertical: 3,
    paddingHorizontal: 8,
    borderRadius: 6,
  },
  advisoryTagText: { color: '#D97706', fontSize: 10 },
  advisoryDesc: { color: COLORS.textMuted, fontSize: 12, lineHeight: 18 },
  advisoryList: { gap: 8 },
  advisoryItem: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  advisoryItemIcon: { fontSize: 14 },
  advisoryItemText: { color: '#FDE68A', fontSize: 13 },
  // Ethics
  ethicsCard: {
    backgroundColor: '#0C1A3A',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#1E3A6E',
    padding: 16,
    gap: 8,
  },
  ethicsTitle: { color: '#60A5FA', fontSize: 13, fontWeight: '600' },
  ethicsText: { color: COLORS.textMuted, fontSize: 12, lineHeight: 18 },
});
