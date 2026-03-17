document.addEventListener('DOMContentLoaded', function() {
    loadFeatureImportance();
    loadTopRiskEmployees();
});

function loadFeatureImportance() {
    fetch('/api/feature_importance')
        .then(response => response.json())
        .then(data => {
            const ctx = document.getElementById('featureChart').getContext('2d');
            const labels = data.map(item => item.feature);
            const values = data.map(item => item.importance_mean);
            
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Importance',
                        data: values,
                        backgroundColor: 'rgba(75, 192, 192, 0.6)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    indexAxis: 'y',
                    scales: {
                        x: {
                            beginAtZero: true
                        }
                    }
                }
            });
        });
}

function loadTopRiskEmployees() {
    fetch('/api/top_risk_employees')
        .then(response => response.json())
        .then(data => {
            console.log('Top risk employees data:', data);
            const tbody = document.querySelector('#riskTable tbody');
            data.forEach(employee => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td><span onclick="showEmployeeDetails(${employee.EmpID})" style="cursor:pointer; color:blue; text-decoration:underline">${employee.Employee_Name}</span></td>
                    <td>${employee.Department || ''}</td>
                    <td>${employee.Position || ''}</td>
                    <td>${(employee.risk_of_leaving * 100).toFixed(2)}%</td>
                `;
                tbody.appendChild(row);
            });
        });
}

function showEmployeeDetails(empId) {
    console.log('Clicked employee ID:', empId);
    fetch(`/api/employee/${empId}`)
        .then(response => {
            console.log('Response status:', response.status);
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('Data received:', data);
            document.getElementById('employeeName').textContent = data.name;
            
            const ctx = document.getElementById('comparisonChart').getContext('2d');
            const labels = data.comparison.map(item => item.Variable);
            const values = data.comparison.map(item => item.Ecart_pct);
            const colors = data.comparison.map(item => item.Color);
            
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Difference from Average (%)',
                        data: values,
                        backgroundColor: colors,
                        borderColor: colors,
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            // Populate table
            const tbody = document.querySelector('#comparisonTable tbody');
            tbody.innerHTML = '';
            data.comparison.forEach(item => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${item.Variable}</td>
                    <td>${item.Employe.toFixed(2)}</td>
                    <td>${item.Moyenne.toFixed(2)}</td>
                    <td style="color: ${item.Color}">${item.Ecart_pct.toFixed(1)}%</td>
                `;
                tbody.appendChild(row);
            });
            
            document.getElementById('employeeModal').style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error loading employee details: ' + error.message);
        });
}

// Close modal
document.querySelector('.close').onclick = function() {
    document.getElementById('employeeModal').style.display = 'none';
}

window.onclick = function(event) {
    if (event.target == document.getElementById('employeeModal')) {
        document.getElementById('employeeModal').style.display = 'none';
    }
}