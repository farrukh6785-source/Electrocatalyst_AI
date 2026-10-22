from unit_conversions import to_current_density

print(to_current_density([0.005], "A", 0.5))
print(to_current_density([5], "mA", 0.5))
print(to_current_density([0.01], "A/cm²"))
print(to_current_density([10], "mA/cm²"))
