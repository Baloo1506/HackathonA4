import { Employee } from './types';

export function parseCSV(text: string): Employee[] {
  const lines = text.trim().split('\n');
  if (lines.length < 2) return [];

  const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));

  return lines.slice(1).map(line => {
    // Handle quoted fields with commas inside
    const values: string[] = [];
    let inQuotes = false;
    let current = '';
    for (const char of line) {
      if (char === '"') { inQuotes = !inQuotes; continue; }
      if (char === ',' && !inQuotes) { values.push(current.trim()); current = ''; continue; }
      current += char;
    }
    values.push(current.trim());

    const row: Record<string, string> = {};
    headers.forEach((h, i) => { row[h] = values[i] ?? ''; });

    return {
      anonymized_id: row['anonymized_id'] || '',
      Department: row['Department'] || '',
      Position: row['Position'] || '',
      risk_score: parseFloat(row['risk_score']) || 0,
      risk_level: (row['risk_level'] as Employee['risk_level']) || 'Low',
      top_reason_1: row['top_reason_1'] || '',
      top_reason_2: row['top_reason_2'] || '',
      top_reason_3: row['top_reason_3'] || '',
      recommended_action: row['recommended_action'] || '',
      nlp_advisory: row['nlp_advisory'] || '',
    };
  }).filter(e => e.anonymized_id !== '');
}
