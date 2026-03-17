import React, { useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { Employee, COLORS, RISK_COLOR } from '../types';

interface Props {
  employees: Employee[];
  onNavigateToEmployees: () => void;
}

interface StatCardProps {
  label: string;
  value: number | string;
  color: string;
  sub?: string;
}

function StatCard({ label, value, color, sub }: StatCardProps) {
  return (
    <View style={[styles.statCard, { borderTopColor: color }]}>
      <Text style={[styles.statValue, { color }]}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
      {sub && <Text style={styles.statSub}>{sub}</Text>}
    </View>
  );
}

export default function DashboardScreen({ employees, onNavigateToEmployees }: Props) {
  const stats = useMemo(() => {
    const total = employees.length;
    const high = employees.filter(e => e.risk_level === 'High').length;
    const medium = employees.filter(e => e.risk_level === 'Medium').length;
    const low = employees.filter(e => e.risk_level === 'Low').length;
    const avgScore = total > 0
      ? (employees.reduce((s, e) => s + e.risk_score, 0) / total * 100).toFixed(1)
      : '0.0';

    // Department risk rates
    const deptMap: Record<string, { total: number; high: number }> = {};
    employees.forEach(e => {
      if (!deptMap[e.Department]) deptMap[e.Department] = { total: 0, high: 0 };
      deptMap[e.Department].total++;
      if (e.risk_level === 'High') deptMap[e.Department].high++;
    });
    const deptRisk = Object.entries(deptMap)
      .map(([dept, d]) => ({ dept, rate: d.total > 0 ? d.high / d.total : 0, total: d.total }))
      .sort((a, b) => b.rate - a.rate)
      .slice(0, 5);

    return { total, high, medium, low, avgScore, deptRisk };
  }, [employees]);

  const highPct = stats.total ? (stats.high / stats.total) * 100 : 0;
  const medPct = stats.total ? (stats.medium / stats.total) * 100 : 0;
  const lowPct = stats.total ? (stats.low / stats.total) * 100 : 0;

  const topHighRisk = employees
    .filter(e => e.risk_level === 'High')
    .sort((a, b) => b.risk_score - a.risk_score)
    .slice(0, 5);

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.title}>Risk Dashboard</Text>
          <Text style={styles.subtitle}>Vue d'ensemble des employés actifs</Text>
        </View>
        <View style={styles.headerBadge}>
          <Text style={styles.headerBadgeText}>IA Explicable</Text>
        </View>
      </View>

      {/* KPI Cards */}
      <View style={styles.kpiRow}>
        <StatCard label="Employés actifs" value={stats.total} color={COLORS.accent} sub="total" />
        <StatCard label="Risque Élevé" value={stats.high} color={COLORS.high} sub={`${highPct.toFixed(0)}%`} />
        <StatCard label="Risque Moyen" value={stats.medium} color={COLORS.medium} sub={`${medPct.toFixed(0)}%`} />
        <StatCard label="Risque Faible" value={stats.low} color={COLORS.low} sub={`${lowPct.toFixed(0)}%`} />
        <StatCard label="Score moyen" value={`${stats.avgScore}%`} color={COLORS.textMuted} sub="risque moy." />
      </View>

      <View style={styles.row}>
        {/* Distribution bar */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Distribution du risque</Text>
          <View style={styles.distBar}>
            {highPct > 0 && (
              <View style={[styles.distSegment, { flex: highPct, backgroundColor: COLORS.high }]} />
            )}
            {medPct > 0 && (
              <View style={[styles.distSegment, { flex: medPct, backgroundColor: COLORS.medium }]} />
            )}
            {lowPct > 0 && (
              <View style={[styles.distSegment, { flex: lowPct, backgroundColor: COLORS.low }]} />
            )}
          </View>
          <View style={styles.legend}>
            {[
              { label: 'Élevé', color: COLORS.high, pct: highPct },
              { label: 'Moyen', color: COLORS.medium, pct: medPct },
              { label: 'Faible', color: COLORS.low, pct: lowPct },
            ].map(item => (
              <View key={item.label} style={styles.legendItem}>
                <View style={[styles.legendDot, { backgroundColor: item.color }]} />
                <Text style={styles.legendText}>{item.label} ({item.pct.toFixed(0)}%)</Text>
              </View>
            ))}
          </View>
        </View>

        {/* Top departments */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Risque par département</Text>
          {stats.deptRisk.map(({ dept, rate, total }) => (
            <View key={dept} style={styles.deptRow}>
              <Text style={styles.deptName} numberOfLines={1}>{dept}</Text>
              <View style={styles.deptBarBg}>
                <View
                  style={[
                    styles.deptBarFill,
                    {
                      width: `${rate * 100}%` as any,
                      backgroundColor: rate > 0.4 ? COLORS.high : rate > 0.2 ? COLORS.medium : COLORS.low,
                    },
                  ]}
                />
              </View>
              <Text style={styles.deptPct}>{(rate * 100).toFixed(0)}%</Text>
            </View>
          ))}
        </View>
      </View>

      {/* Top high-risk employees */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Text style={styles.cardTitle}>🔴 Employés à risque élevé prioritaires</Text>
          <TouchableOpacity onPress={onNavigateToEmployees}>
            <Text style={styles.seeAll}>Voir tous →</Text>
          </TouchableOpacity>
        </View>
        <View style={styles.tableHeader}>
          <Text style={[styles.tableCell, styles.tableHeaderText, { flex: 1.2 }]}>ID</Text>
          <Text style={[styles.tableCell, styles.tableHeaderText, { flex: 2 }]}>Département</Text>
          <Text style={[styles.tableCell, styles.tableHeaderText, { flex: 2 }]}>Poste</Text>
          <Text style={[styles.tableCell, styles.tableHeaderText, { flex: 1.5 }]}>Raison principale</Text>
          <Text style={[styles.tableCell, styles.tableHeaderText, { flex: 1, textAlign: 'right' }]}>Score</Text>
        </View>
        {topHighRisk.map((emp, i) => (
          <View key={emp.anonymized_id} style={[styles.tableRow, i % 2 === 1 && styles.tableRowAlt]}>
            <Text style={[styles.tableCell, styles.tableId, { flex: 1.2 }]}>{emp.anonymized_id}</Text>
            <Text style={[styles.tableCell, styles.tableCellText, { flex: 2 }]} numberOfLines={1}>{emp.Department}</Text>
            <Text style={[styles.tableCell, styles.tableCellText, { flex: 2 }]} numberOfLines={1}>{emp.Position}</Text>
            <Text style={[styles.tableCell, styles.tableCellMuted, { flex: 1.5 }]} numberOfLines={1}>
              {emp.top_reason_1.replace(/_/g, ' ')}
            </Text>
            <View style={[styles.tableCell, { flex: 1, alignItems: 'flex-end' }]}>
              <View style={styles.scorePill}>
                <Text style={styles.scorePillText}>{(emp.risk_score * 100).toFixed(0)}%</Text>
              </View>
            </View>
          </View>
        ))}
        {topHighRisk.length === 0 && (
          <Text style={styles.emptyText}>Aucun employé à risque élevé</Text>
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: 28, gap: 20 },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: { color: COLORS.text, fontSize: 24, fontWeight: '700' },
  subtitle: { color: COLORS.textMuted, fontSize: 14, marginTop: 4 },
  headerBadge: {
    backgroundColor: '#1E1B4B',
    borderWidth: 1,
    borderColor: '#4338CA',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 20,
  },
  headerBadgeText: { color: '#818CF8', fontSize: 12, fontWeight: '500' },
  // KPI
  kpiRow: { flexDirection: 'row', gap: 14 },
  statCard: {
    flex: 1,
    backgroundColor: COLORS.surface,
    borderRadius: 10,
    padding: 16,
    borderTopWidth: 3,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  statValue: { fontSize: 28, fontWeight: '700', marginBottom: 4 },
  statLabel: { color: COLORS.text, fontSize: 13, fontWeight: '500' },
  statSub: { color: COLORS.textMuted, fontSize: 11, marginTop: 2 },
  // Row
  row: { flexDirection: 'row', gap: 14 },
  card: {
    flex: 1,
    backgroundColor: COLORS.surface,
    borderRadius: 10,
    padding: 20,
    borderWidth: 1,
    borderColor: COLORS.border,
    gap: 12,
  },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  cardTitle: { color: COLORS.text, fontSize: 15, fontWeight: '600' },
  seeAll: { color: COLORS.accent, fontSize: 13 },
  // Distribution
  distBar: {
    flexDirection: 'row',
    height: 18,
    borderRadius: 9,
    overflow: 'hidden',
    gap: 2,
  },
  distSegment: { height: '100%' },
  legend: { flexDirection: 'row', gap: 16 },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  legendDot: { width: 8, height: 8, borderRadius: 4 },
  legendText: { color: COLORS.textMuted, fontSize: 12 },
  // Dept
  deptRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  deptName: { color: COLORS.textMuted, fontSize: 12, width: 90 },
  deptBarBg: {
    flex: 1,
    height: 8,
    backgroundColor: COLORS.border,
    borderRadius: 4,
    overflow: 'hidden',
  },
  deptBarFill: { height: '100%', borderRadius: 4 },
  deptPct: { color: COLORS.textMuted, fontSize: 12, width: 32, textAlign: 'right' },
  // Table
  tableHeader: {
    flexDirection: 'row',
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  tableHeaderText: { color: COLORS.textDim, fontSize: 11, fontWeight: '600', textTransform: 'uppercase' },
  tableRow: { flexDirection: 'row', paddingVertical: 10, alignItems: 'center' },
  tableRowAlt: { backgroundColor: '#ffffff05', borderRadius: 6 },
  tableCell: { paddingHorizontal: 4 },
  tableId: { color: COLORS.accent, fontSize: 12, fontWeight: '600' },
  tableCellText: { color: COLORS.text, fontSize: 13 },
  tableCellMuted: { color: COLORS.textMuted, fontSize: 12 },
  scorePill: {
    backgroundColor: '#450A0A',
    borderWidth: 1,
    borderColor: COLORS.high,
    paddingVertical: 2,
    paddingHorizontal: 8,
    borderRadius: 12,
  },
  scorePillText: { color: COLORS.high, fontSize: 11, fontWeight: '700' },
  emptyText: { color: COLORS.textDim, fontSize: 13, textAlign: 'center', padding: 12 },
});
