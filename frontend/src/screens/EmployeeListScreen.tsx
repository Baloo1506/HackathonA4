import React, { useState, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
} from 'react-native';
import { Employee, COLORS, RISK_COLOR, RISK_BG, FEATURE_LABELS } from '../types';

interface Props {
  employees: Employee[];
  onSelect: (emp: Employee) => void;
}

type Filter = 'All' | 'High' | 'Medium' | 'Low';

const FILTERS: Filter[] = ['All', 'High', 'Medium', 'Low'];
const FILTER_LABEL: Record<Filter, string> = {
  All: 'Tous',
  High: '🔴 Élevé',
  Medium: '🟠 Moyen',
  Low: '🟢 Faible',
};

export default function EmployeeListScreen({ employees, onSelect }: Props) {
  const [filter, setFilter] = useState<Filter>('All');
  const [search, setSearch] = useState('');

  const filtered = useMemo(() => {
    return employees
      .filter(e => filter === 'All' || e.risk_level === filter)
      .filter(e => {
        if (!search) return true;
        const q = search.toLowerCase();
        return (
          e.anonymized_id.toLowerCase().includes(q) ||
          e.Department.toLowerCase().includes(q) ||
          e.Position.toLowerCase().includes(q)
        );
      })
      .sort((a, b) => b.risk_score - a.risk_score);
  }, [employees, filter, search]);

  const counts = useMemo(() => ({
    All: employees.length,
    High: employees.filter(e => e.risk_level === 'High').length,
    Medium: employees.filter(e => e.risk_level === 'Medium').length,
    Low: employees.filter(e => e.risk_level === 'Low').length,
  }), [employees]);

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.title}>Liste des employés</Text>
          <Text style={styles.subtitle}>{filtered.length} résultats</Text>
        </View>
        <View style={styles.searchBox}>
          <Text style={styles.searchIcon}>🔍</Text>
          <TextInput
            style={styles.searchInput}
            placeholder="Rechercher ID, département..."
            placeholderTextColor={COLORS.textDim}
            value={search}
            onChangeText={setSearch}
          />
        </View>
      </View>

      {/* Filter tabs */}
      <View style={styles.filterRow}>
        {FILTERS.map(f => (
          <TouchableOpacity
            key={f}
            style={[styles.filterTab, filter === f && styles.filterTabActive]}
            onPress={() => setFilter(f)}
          >
            <Text style={[styles.filterLabel, filter === f && styles.filterLabelActive]}>
              {FILTER_LABEL[f]}
            </Text>
            <View style={[styles.filterCount, filter === f && styles.filterCountActive]}>
              <Text style={[styles.filterCountText, filter === f && styles.filterCountTextActive]}>
                {counts[f]}
              </Text>
            </View>
          </TouchableOpacity>
        ))}
      </View>

      {/* Employee list */}
      <ScrollView style={styles.list} contentContainerStyle={styles.listContent}>
        {filtered.map(emp => (
          <EmployeeCard key={emp.anonymized_id} employee={emp} onPress={() => onSelect(emp)} />
        ))}
        {filtered.length === 0 && (
          <View style={styles.empty}>
            <Text style={styles.emptyIcon}>🔍</Text>
            <Text style={styles.emptyText}>Aucun employé trouvé</Text>
          </View>
        )}
      </ScrollView>
    </View>
  );
}

function EmployeeCard({ employee: emp, onPress }: { employee: Employee; onPress: () => void }) {
  const riskColor = RISK_COLOR(emp.risk_level);
  const riskBg = RISK_BG(emp.risk_level);
  const reasons = [emp.top_reason_1, emp.top_reason_2, emp.top_reason_3].filter(Boolean);
  const advisories = emp.nlp_advisory ? emp.nlp_advisory.split(' | ').filter(Boolean) : [];

  return (
    <TouchableOpacity style={styles.card} onPress={onPress} activeOpacity={0.7}>
      <View style={styles.cardTop}>
        {/* Left: ID + dept + position */}
        <View style={styles.cardInfo}>
          <View style={styles.cardIdRow}>
            <Text style={styles.cardId}>{emp.anonymized_id}</Text>
            {advisories.length > 0 && (
              <View style={styles.advisoryPill}>
                <Text style={styles.advisoryPillText}>⚠ {advisories.length} alerte{advisories.length > 1 ? 's' : ''}</Text>
              </View>
            )}
          </View>
          <Text style={styles.cardDept}>{emp.Department}</Text>
          <Text style={styles.cardPosition}>{emp.Position}</Text>
        </View>

        {/* Right: Risk badge + score */}
        <View style={styles.cardRight}>
          <View style={[styles.riskBadge, { backgroundColor: riskBg, borderColor: riskColor }]}>
            <Text style={[styles.riskBadgeText, { color: riskColor }]}>{emp.risk_level}</Text>
          </View>
          <Text style={[styles.riskScore, { color: riskColor }]}>
            {(emp.risk_score * 100).toFixed(0)}%
          </Text>
        </View>
      </View>

      {/* Score bar */}
      <View style={styles.scoreBarBg}>
        <View
          style={[
            styles.scoreBarFill,
            { width: `${emp.risk_score * 100}%` as any, backgroundColor: riskColor },
          ]}
        />
      </View>

      {/* Reasons */}
      {reasons.length > 0 && (
        <View style={styles.reasonsRow}>
          {reasons.slice(0, 3).map((r, i) => (
            <View key={i} style={styles.reasonChip}>
              <Text style={styles.reasonText}>
                {FEATURE_LABELS[r] ?? r.replace(/_/g, ' ')}
              </Text>
            </View>
          ))}
        </View>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 28,
    paddingBottom: 16,
  },
  title: { color: COLORS.text, fontSize: 22, fontWeight: '700' },
  subtitle: { color: COLORS.textMuted, fontSize: 13, marginTop: 3 },
  searchBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    gap: 8,
    minWidth: 260,
  },
  searchIcon: { fontSize: 14 },
  searchInput: { color: COLORS.text, fontSize: 13, flex: 1, outlineStyle: 'none' } as any,
  // Filters
  filterRow: {
    flexDirection: 'row',
    paddingHorizontal: 28,
    gap: 8,
    marginBottom: 16,
  },
  filterTab: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: COLORS.border,
    backgroundColor: 'transparent',
  },
  filterTabActive: {
    backgroundColor: COLORS.surface,
    borderColor: COLORS.accent,
  },
  filterLabel: { color: COLORS.textMuted, fontSize: 13, fontWeight: '500' },
  filterLabelActive: { color: COLORS.text },
  filterCount: {
    backgroundColor: COLORS.border,
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: 10,
  },
  filterCountActive: { backgroundColor: COLORS.accent },
  filterCountText: { color: COLORS.textMuted, fontSize: 11, fontWeight: '600' },
  filterCountTextActive: { color: '#fff' },
  // List
  list: { flex: 1 },
  listContent: { paddingHorizontal: 28, paddingBottom: 28, gap: 10 },
  // Card
  card: {
    backgroundColor: COLORS.surface,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 16,
    gap: 10,
  },
  cardTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
  cardInfo: { flex: 1, gap: 3 },
  cardIdRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  cardId: { color: COLORS.accent, fontSize: 13, fontWeight: '700' },
  advisoryPill: {
    backgroundColor: '#431407',
    borderWidth: 1,
    borderColor: COLORS.medium,
    paddingVertical: 1,
    paddingHorizontal: 7,
    borderRadius: 10,
  },
  advisoryPillText: { color: COLORS.medium, fontSize: 10 },
  cardDept: { color: COLORS.text, fontSize: 13, fontWeight: '500' },
  cardPosition: { color: COLORS.textMuted, fontSize: 12 },
  cardRight: { alignItems: 'flex-end', gap: 4 },
  riskBadge: {
    paddingVertical: 3,
    paddingHorizontal: 10,
    borderRadius: 12,
    borderWidth: 1,
  },
  riskBadgeText: { fontSize: 11, fontWeight: '700' },
  riskScore: { fontSize: 20, fontWeight: '700' },
  // Score bar
  scoreBarBg: {
    height: 4,
    backgroundColor: COLORS.border,
    borderRadius: 2,
    overflow: 'hidden',
  },
  scoreBarFill: { height: '100%', borderRadius: 2 },
  // Reasons
  reasonsRow: { flexDirection: 'row', gap: 6, flexWrap: 'wrap' },
  reasonChip: {
    backgroundColor: '#1E293B',
    borderWidth: 1,
    borderColor: COLORS.border,
    paddingVertical: 3,
    paddingHorizontal: 8,
    borderRadius: 6,
  },
  reasonText: { color: COLORS.textMuted, fontSize: 11 },
  // Empty
  empty: { alignItems: 'center', paddingVertical: 60, gap: 12 },
  emptyIcon: { fontSize: 36 },
  emptyText: { color: COLORS.textMuted, fontSize: 14 },
});
